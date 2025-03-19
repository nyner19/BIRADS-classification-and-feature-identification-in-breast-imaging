import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms
from data.basic_datamodule import BasicDataModule
from data.binary_dataset import TumorDataset
from models.utils import MyRotateTransform
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from torch import Tensor
from torchvision import models

# 设置随机种子，保证实验的可重复性
seed_everything(42, workers=True)
torch.manual_seed(42)
np.random.seed(42)


class ClassifierLightning(pl.LightningModule):
    def __init__(
            self,
            num_classes: int = 6,
            lr: float = 0.0001,
            weight_decay: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")

        # 基础模型
        self._model: nn.Module = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )

        # 获取特征输出维度
        self.features_dim = self._model.classifier[1].in_features  # 1280

        # 修改分类器 - 增加中间层维度
        self._model.classifier = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # 修改额外的特征提取层 - 渐进降维
        self.additional_features = nn.Sequential(
            nn.Linear(self.features_dim, 1024),  # 从512增加到1024
            nn.ReLU(),
            nn.Linear(1024, 512),  # 从256增加到512
            nn.ReLU(),
            nn.Linear(512, 256),  # 从128增加到256
            nn.ReLU(),
        )

        # 修改最终分类层 - 适应新的拼接维度
        self.final_classifier = nn.Linear(self.features_dim + 256, num_classes)  # 1280 + 256

        # 重新添加训练指标
        self._train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._train_weighted_acc = torchmetrics.Accuracy(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self._train_f1_score = torchmetrics.F1Score(
            task="multiclass", average="macro", num_classes=num_classes
        )

        # 重新添加验证指标
        self._val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._val_weighted_acc = torchmetrics.Accuracy(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self._val_f1_score = torchmetrics.F1Score(
            task="multiclass", average="macro", num_classes=num_classes
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # 获取特征
        features = self._model.features(x)
        # 使用正确的池化操作
        features = self._model.avgpool(features)
        # 展平特征
        features = torch.flatten(features, 1)
        # 获取额外特征
        additional_features = self.additional_features(features)
        # 拼接特征
        combined_features = torch.cat((features, additional_features), dim=1)
        # 最终分类
        output = self.final_classifier(combined_features)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = train_transforms(x)  # 应用数据增强
        y_pred = self.forward(x)
        y_pred = torch.squeeze(y_pred)
        loss = F.cross_entropy(y_pred, y)

        # 记录训练指标
        self.log("train_loss", loss, on_epoch=False, prog_bar=True)
        self._train_acc(y_pred, y)
        self.log("train_acc", self._train_acc, on_epoch=False, prog_bar=True)
        self._train_weighted_acc(y_pred, y)
        self.log("train_weighted_acc", self._train_weighted_acc, on_epoch=False, prog_bar=True)
        self._train_f1_score(y_pred, y)
        self.log("train_f1_score", self._train_f1_score, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y_pred = torch.squeeze(y_pred)
        loss = F.cross_entropy(y_pred, y)

        # 记录验证指标
        self.log("val_loss", loss, prog_bar=True)
        self._val_acc(y_pred, y)
        self.log("val_acc", self._val_acc, prog_bar=True)
        self._val_weighted_acc(y_pred, y)
        self.log("val_weighted_acc", self._val_weighted_acc, prog_bar=True)
        self._val_f1_score(y_pred, y)
        self.log("val_f1_score", self._val_f1_score, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self(x.float())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=150,
            eta_min=1e-6
        )

        return [optimizer], [scheduler]


# 图像预处理：调整大小和标准化
transformations = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

# 定义训练时的数据增强变换
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(20),  # 随机旋转，角度范围为 [-20, 20]
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪后调整大小
        MyRotateTransform(angles=[0, 180]),  # 自定义旋转变换
    ]
)

# 设置模型保存时的文件名
checkpoint_name = "PAGE_2500_V2S_demo01"
BATCH_SIZE = 36  # 批量大小
DATASET_PATH = "/root/lanyun-tmp/BI_RADS/BI_RADS/dataset"  # 数据集路径
dataset = ["BUS"]  # 数据集的名称

# 初始化模型
model = ClassifierLightning()

# 定义数据模块
tumor_datamodule = BasicDataModule(
    dataset_class=TumorDataset,
    dataset_init_kwargs=dict(
        dataset_path=DATASET_PATH,
        data_transforms=transformations,
        folders=dataset
    ),
    train_val_test_fractions=(0.80, 0.20, 0.0),  # 数据集划分比例
    batch_size=BATCH_SIZE,
    num_workers=4,
    kfold_splits=5,
    n_split_kfold=0,
)

# 定义提前停止回调
early_stopping_callback = EarlyStopping(
    monitor="val_f1_score",
    patience=25,
    mode="max",
    verbose=True
)

# 定义模型检查点回调
checkpoint_callback = ModelCheckpoint(
    dirpath=f"checkpoints/{checkpoint_name}",
    save_top_k=1,
    monitor="val_weighted_acc",
    mode="max",
    auto_insert_metric_name=True,
    filename="{epoch}-{val_weighted_acc:.3f}-{val_f1_score:.3f}",
)

# 配置训练器
trainer = Trainer(
    max_epochs=150,
    deterministic=True,
    callbacks=[
        early_stopping_callback,
        checkpoint_callback,
        StochasticWeightAveraging(swa_lrs=1e-2),
    ],
)

# 主函数
if __name__ == "__main__":
    trainer.fit(model, datamodule=tumor_datamodule)
    torch.save(model.state_dict(), f"{checkpoint_name}.pt")