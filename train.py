import hydra
import os

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from omegaconf import DictConfig, OmegaConf

from model.WaterLevelModel import WaterLevelModel
from DataModule import WaterLevelDataModule

@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(args: DictConfig):
    L.seed_everything(args.seed)
    model = WaterLevelModel(**args.model,task=args.task, dataset=args.dataset)
    dm = WaterLevelDataModule(**args.dataset,task=args.task)
    logger_tb = TensorBoardLogger("lightning_logs",name=str(args.task.logs_dir))
    cvs_logger = CSVLogger("lightning_logs",name=str(args.task.logs_dir),version=str(logger_tb.version) + "_csv")
    trainer = L.Trainer(enable_checkpointing=True,**args.trainer,logger=[logger_tb,cvs_logger])
    
    trainer.fit(model, datamodule=dm)
    
    config_path = os.path.join(logger_tb.log_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config=args, f=f)
    
if __name__ == "__main__":
    main()