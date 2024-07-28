from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import os
import yaml
import xarray as xr
from scipy.stats import kstest

class WaterLevelDataModule(LightningDataModule):
    def __init__(self,task : DictConfig, **hparams : DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["task"] = task.name
        self.batch_size = self.hparams["batch_size"]
        self.num_workers = self.hparams["num_workers"]
        self.data_dir = self.hparams["data_dir"]
        
    def setup(self, stage=None):
        if self.hparams["manual_split"]:
            self.train_dataset = WaterLevelDataset(root=self.data_dir,split="train",task=self.hparams["task"],steps=self.hparams["steps"],threshold=self.hparams["threshold"])
            self.val_dataset = WaterLevelDataset(root=self.data_dir,split="eval",task=self.hparams["task"],steps=self.hparams["steps"],threshold=self.hparams["threshold"])
        else:
            dataset = WaterLevelDataset(root=self.data_dir,split="all",task=self.hparams["task"],steps=self.hparams["steps"],threshold=self.hparams["threshold"])
            train_size = int(0.8 * len(dataset))
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        if self.hparams["task"] == "classification":
            print("Zero samples in Eval: ", len([x for x in [x[1] for x in self.val_dataset] if x == 0]))
            print("Positive samples in Eval: ", len([x for x in [x[1] for x in self.val_dataset] if x == 1]))
            print("Negative samples in Eval: ", len([x for x in [x[1] for x in self.val_dataset] if x == 2]))
        elif self.hparams["task"] == "regression":
            print("Mean of Train: ", np.mean([x[1] for x in self.train_dataset]))
            print("Mean of Eval: ", np.mean([x[1] for x in self.val_dataset]))
            print("Std of Train: ", np.std([x[1] for x in self.train_dataset]))
            print("Std of Eval: ", np.std([x[1] for x in self.val_dataset]))
            print(kstest([x[1] for x in self.train_dataset], [x[1] for x in self.val_dataset]))
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=True,persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=2,num_workers=self.num_workers,shuffle=False,persistent_workers=True,drop_last=True)
        
class WaterLevelDataset(Dataset):
    
    classColors = [(255, 0, 0), (255, 255, 255), (0, 0, 255)]
    classColors = np.array(classColors)
    
    def __init__(self, root : str,split : str, task : str,threshold : float = 0.02,steps: int = 4):
        self.root = root
        self.tiles = []
        self.targets = []
        self.split = split
        self.task = task.name
        df = pd.read_csv(
            os.path.join(root,"dataset.csv"),
            header=0,
            usecols=["name", "coordinates", "category", "downloaded", "date", "split"],
            index_col=["name"],
            sep=",",
        )
        if steps < 5:
            self.start = 4 - steps
            self.end = 4
        else:
            raise ValueError("Steps must be at most 4")
        df = df[(df["downloaded"] == 1) & (df["split"] == self.split)] if self.split != "all" else df[df["downloaded"] == 1]
        for name, row in df.iterrows():
            folder = os.path.join(root,"data", name.__str__())
            if not os.path.isdir(folder):
                raise ValueError(f"Folder {folder} does not exist")
            for subdir in os.listdir(os.path.join(folder, "tiles")):
                if not os.path.exists(
                    os.path.join(folder, "tiles", subdir, "metadata.yaml")
                ) or not os.path.exists(
                    os.path.join(folder, "tiles", subdir, "features.nc")
                ):
                    continue
                tile = os.path.join(folder, "tiles", subdir, "features.nc")
                if xr.open_dataarray(tile,decode_coords="all").squeeze().shape[-2:] != (256, 256):
                    print("File " + tile+" has shape " + xr.open_dataarray(tile,decode_coords="all").squeeze().shape.__str__())
                    continue
                change = yaml.load(
                    open(os.path.join(folder, "tiles", subdir, "metadata.yaml"), "r"),
                    Loader=yaml.FullLoader,
                )["change"]
                self.tiles.append(tile)
                match self.task:
                    case "classification":
                        if change > threshold:
                            self.targets.append(1)
                        elif change < -threshold:
                            self.targets.append(2)
                        else:
                            self.targets.append(0)
                    case "regression":
                        self.targets.append(change) if abs(change) > threshold else self.targets.append(0.0)
                    case "segmentation":
                        mask = os.path.join(folder, "tiles", subdir, "mask.nc")
                        self.targets.append(mask)

    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile = xr.open_dataarray(self.tiles[idx],decode_coords="all").squeeze()
        if (tile.shape[-2:] != (256, 256)):
            raise ValueError(f"File {tile} has shape {tile.shape}")
        tile = tile.to_numpy()[self.start:self.end].reshape(-1,256,256) #TODO: Check if this is correct
        if self.task == "segmentation":
            label = xr.open_dataset(self.targets[idx],decode_coords="all")["water_change"].squeeze().to_numpy() + 1 #shifts from -1,0,1 to 0,1,2
        else:
            label = np.array(self.targets[idx])
        return tile, label
    
    @classmethod
    def colorizePrediction(cls, prediction):
        return cls.classColors[np.argmax(prediction[0].detach().cpu().numpy(),axis=0).astype("uint8")].astype("uint8")
    #Image.fromarray(colors[].astype("uint8")).show()

    @classmethod
    def colorizeTarget(cls, target):
        return cls.classColors[target[0].detach().cpu().numpy().astype("uint8")].astype("uint8")
    
if __name__ == "__main__":
    DataModule=WaterLevelDataModule(data_dir="data\\NDWI256",batch_size=1,num_workers=0,manual_split=False,task="segmentation",steps=4,threshold=0.0001)
    DataModule.setup()
    train_loader = DataModule.train_dataloader()
    train_loader = iter(train_loader)
    print(next(train_loader))
    
    
#from PIL import Image
#import numpy as np
#classColors = [(255, 0, 0), (255, 255, 255), (0, 0, 255)]
#classColors = np.array(classColors)
#predictions = Image.fromarray(classColors[np.argmax(out[0].detach().cpu().numpy(),axis=0).astype("uint8")].astype("uint8"))
#labels = Image.fromarray(classColors[y[0].detach().cpu().numpy().astype("uint8")].astype("uint8"))
#predictions.show()
#labels.show()