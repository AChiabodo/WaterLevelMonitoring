import hydra
import os

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from omegaconf import DictConfig, OmegaConf

from model.WaterLevelModel import WaterLevelModel
from DataModule import WaterLevelDataModule

@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(args: DictConfig):
    L.seed_everything(args.seed)
    model = WaterLevelModel(**args.model,task=args.task, dataset=args.dataset)
    dm = WaterLevelDataModule(**args.dataset,task=args.task)
    logger = TensorBoardLogger("lightning_logs",name=str(args.task.name))
    trainer = L.Trainer(enable_checkpointing=False,**args.trainer,logger=logger)
    
    trainer.fit(model, datamodule=dm)
    
    config_path = os.path.join(logger.log_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config=args, f=f)
    
if __name__ == "__main__":
    main()