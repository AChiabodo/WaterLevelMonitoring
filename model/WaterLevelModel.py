import timm
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError, R2Score
from lightning import LightningModule
from omegaconf import DictConfig
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

SUPPORTED_TASKS = ["classification", "regression"]

class WaterLevelModel(LightningModule):
    def __init__(self, **hparams : DictConfig):
        super().__init__()
        self.save_hyperparameters()
        
        if self.hparams["task"] not in SUPPORTED_TASKS:
            raise ValueError(f"Task {self.hparams['task']} not supported. Supported tasks are {SUPPORTED_TASKS}")
        
        self.model = timm.create_model(
                self.hparams["model_name"],
                pretrained=self.hparams["pretrained"],
                num_classes=self.hparams["num_classes"] if self.hparams["task"] == "classification" else 1,
                in_chans=self.hparams["in_chans"],
        )
        
        self.loss = FocalLoss(mode="multiclass", alpha=self.hparams["alpha"], gamma=self.hparams["gamma"]) if self.hparams["task"] == "classification" else nn.HuberLoss()
        self.train_acc = Accuracy(task="multiclass",num_classes=3) if self.hparams["task"] == "classification" else MeanAbsoluteError()
        self.val_acc = Accuracy(task="multiclass",num_classes=3) if self.hparams["task"] == "classification" else MeanAbsoluteError()
        self.f1 = F1Score(task="multiclass",num_classes=3,average="macro") if self.hparams["task"] == "classification" else None
        self.r2 = R2Score() if self.hparams["task"] == "regression" else None
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        if self.hparams["task"] == "classification":
            self.train_acc(out.argmax(1), y)
            loss = self.loss(out, y)
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        if self.hparams["task"] == "regression":
            self.train_acc(out.float().squeeze(), y.float().squeeze())
            loss = self.loss(out.float().squeeze(1), y.float())
            self.log("train_err", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        if self.hparams["task"] == "classification":
            self.val_acc(out.argmax(1), y)
            self.log("valid_acc", self.val_acc, on_step=False, on_epoch=True)
            self.f1(out.argmax(1), y)
            self.log("valid_f1", self.f1, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams["task"] == "regression":
            self.val_acc(out.float().squeeze(1), y.float())
            self.log("valid_err", self.val_acc, on_step=False, on_epoch=True)
            self.r2(out.float().squeeze(1), y.float())
            self.log("train_r2", self.r2, on_step=False, on_epoch=True, prog_bar=True)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr", lr, on_step=False, on_epoch=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]