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
        
    def setup(self, stage=None):
        if self.hparams["manual_split"]:
            self.train_dataset =    WaterLevelDataset(split="train", hparams=self.hparams)
            self.val_dataset =      WaterLevelDataset(split="eval",  hparams=self.hparams)
        else:
            dataset =               WaterLevelDataset(split="all",   hparams=self.hparams)
            train_size = int(0.8 * len(dataset))
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        
        if self.hparams["task"]["name"] == "classification":
            print("Zero samples in Eval: ", len([x for x in [x[1] for x in self.val_dataset] if x == 0]))
            print("Positive samples in Eval: ", len([x for x in [x[1] for x in self.val_dataset] if x == 1]))
            print("Negative samples in Eval: ", len([x for x in [x[1] for x in self.val_dataset] if x == 2]))
        elif self.hparams["task"]["name"] == "regression":
            print("Mean of Train: ", np.mean([x[1] for x in self.train_dataset]))
            print("Mean of Eval: ", np.mean([x[1] for x in self.val_dataset]))
            print("Std of Train: ", np.std([x[1] for x in self.train_dataset]))
            print("Std of Eval: ", np.std([x[1] for x in self.val_dataset]))
            print(kstest([x[1] for x in self.train_dataset], [x[1] for x in self.val_dataset]))
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.hparams["batch_size"],num_workers=self.hparams["num_workers"],shuffle=True,persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=2,num_workers=self.hparams["num_workers"],shuffle=False,persistent_workers=True,drop_last=True)
        
class WaterLevelDataset(Dataset):
    
    classColors = [(255, 0, 0), (255, 255, 255), (0, 0, 255)]
    classColors = np.array(classColors)
    
    def __init__(self, split : str, hparams : DictConfig):
        self.root = hparams["data_dir"]
        self.tiles = []
        self.targets = []
        self.split = split
        self.task = hparams["task"]["name"]
        self.steps = hparams["task"]["steps"]
        self.bands = hparams["bands"]
        self.patch_size = hparams["patch_size"]
        self.threshold = hparams["threshold"]
        self.focus = hparams["task"]["focus"] if "focus" in hparams["task"] else None
        self.eval_lakes = hparams["eval_lakes"] if len(hparams["eval_lakes"]) > 0 else []
        df = pd.read_csv(
            os.path.join(self.root,"dataset.csv"),
            header=0,
            usecols=["name", "coordinates", "category", "downloaded", "date", "split"],
            index_col=["name"],
            sep=",",
        )
        
        if not self.eval_lakes:
            df = df[(df["downloaded"] == 1) & (df["split"] == self.split)] if self.split != "all" else df[df["downloaded"] == 1]
        else:
            if not all([x in df.index for x in self.eval_lakes]):
                raise ValueError(f"Eval lakes {self.eval_lakes} not in dataset")
            elif self.split == "eval":
                df = df[(df["downloaded"] == 1) & df.index.isin(self.eval_lakes)]
            elif self.split == "train":
                df = df[(df["downloaded"] == 1) & ~df.index.isin(self.eval_lakes)]
        
        for name, row in df.iterrows():
            folder = os.path.join(self.root, name.__str__())
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
                if xr.open_dataarray(tile,decode_coords="all").squeeze().shape[-2:] != (self.patch_size, self.patch_size):
                    print("File " + tile+" has shape " + xr.open_dataarray(tile,decode_coords="all").squeeze().shape.__str__())
                    continue
                change = yaml.load(
                    open(os.path.join(folder, "tiles", subdir, "metadata.yaml"), "r"),
                    Loader=yaml.FullLoader,
                )["change"]
                self.tiles.append(tile)
                match self.task:
                    case "classification":
                        if change > self.threshold:
                            self.targets.append(1)
                        elif change < -self.threshold:
                            self.targets.append(2)
                        else:
                            self.targets.append(0)
                    case "regression":
                        self.targets.append(change) if abs(change) > self.threshold else self.targets.append(0.0)
                    case "segmentation":
                        if self.focus == "temporal":
                            mask = os.path.join(folder, "tiles", subdir, "mask.nc")
                        else:
                            mask = os.path.join(folder, "tiles", subdir, "water.nc")
                        self.targets.append(mask)

    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile = xr.open_dataarray(self.tiles[idx],decode_coords="all")
        tile = tile[self.bands][:,self.steps].squeeze() if tile.ndim == 4 else tile[self.steps].squeeze()
        if (tile.shape[-2:] != (self.patch_size, self.patch_size)):
            raise ValueError(f"File {tile} has shape {tile.shape}")
        tile = tile.to_numpy().reshape(-1,self.patch_size,self.patch_size) #TODO: Check if this is correct
        if self.task == "segmentation":
            if self.focus == "temporal":
                label = xr.open_dataset(self.targets[idx],decode_coords="all")["water_change"].squeeze().to_numpy() + 1 #shifts from -1,0,1 to 0,1,2
            else:
                label = xr.open_dataset(self.targets[idx],decode_coords="all")["start"].squeeze().to_numpy()
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