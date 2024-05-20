import hydra
import lightning as L
from omegaconf import DictConfig
from model.WaterLevelModel import WaterLevelModel
from DataModule import WaterLevelDataModule

@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(args: DictConfig):
    L.seed_everything(args.seed)
    model = WaterLevelModel(**args.model)
    dm = WaterLevelDataModule(**args.dataset)
    trainer = L.Trainer(**args.trainer)
    trainer.fit(model, datamodule=dm)
    
if __name__ == "__main__":
    main()