import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import csv
import pytorch_lightning as pl
import torchvision.models as models
from torch import nn
from tqdm import tqdm

# 设置随机种子
pl.seed_everything(42, workers=True)
torch.manual_seed(42)
np.random.seed(42)

# 定义图像预处理
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class UnlabeledTumorDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))],
            key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]


class ClassifierLightning6(pl.LightningModule):
    def __init__(self, num_classes: int = 6):  # 修改默认类别数为6
        super().__init__()
        # 基础模型
        self._model = models.efficientnet_v2_s(weights=None)
        self.features_dim = self._model.classifier[1].in_features  # 通常是 1280

        # 修改分类器
        self._model.classifier = nn.Sequential(
            nn.Linear(self.features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # 额外的特征提取层
        self.additional_features = nn.Sequential(
            nn.Linear(self.features_dim, 1024),  # 从512增加到1024
            nn.ReLU(),
            nn.Linear(1024, 512),  # 从256增加到512
            nn.ReLU(),
            nn.Linear(512, 256),  # 从128增加到256
            nn.ReLU(),
        )

        # 最终分类层
        self.final_classifier = nn.Linear(self.features_dim + 256, num_classes)

    def forward(self, x):
        # 获取特征
        features = self._model.features(x)
        # 使用正确的池化操作
        features = self._model.avgpool(features)
        # 展开特征
        features = torch.flatten(features, 1)
        # 获取额外特征
        additional_features = self.additional_features(features)
        # 拼接特征
        combined_features = torch.cat((features, additional_features), dim=1)
        # 最终分类
        output = self.final_classifier(combined_features)
        return output


class ClassifierLightning4(pl.LightningModule):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self._model = models.efficientnet_v2_s(weights=None)
        self._model.classifier[1] = nn.Linear(self._model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self._model(x)


if __name__ == "__main__":
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载6分类模型
    print("加载6分类模型...")
    model_6 = ClassifierLightning6()
    try:
        model_6.load_state_dict(torch.load("model/PAGE_2500_V2S_demoFinal.pt"))
        model_6.eval().to(device)
        print("6分类模型加载完成！")
    except Exception as e:
        print(f"加载6分类模型时出错: {e}")

    # 加载6分类测试数据
    print("加载6分类测试集...")
    test_dataset_6 = UnlabeledTumorDataset(
        dataset_path="testB/cla",
        transform=transformations
    )
    test_loader_6 = DataLoader(test_dataset_6, batch_size=1, num_workers=0)
    print("6分类测试集加载完成！")

    # 创建6分类预测结果保存路径
    result_dir = "."
    os.makedirs(result_dir, exist_ok=True)

    # 6分类预测
    csv_path_6 = os.path.join(result_dir, 'cla_pre.csv')
    print(f"保存6分类预测结果到 CSV 文件: {csv_path_6}")
    with open(csv_path_6, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img_name', 'label'])
        with torch.no_grad():
            for image_tensor, image_name in tqdm(test_loader_6, desc="Predicting 6-class", unit="image"):
                image_tensor = image_tensor.to(device)
                pred = model_6(image_tensor)
                class_idx = torch.argmax(pred, dim=1).item()
                writer.writerow([image_name[0], class_idx])  # 包括名称和扩展名
    print("6分类预测完成！")

    # 定义四个模型路径
    model_paths_4 = [
        "model/B5_925.pt",  # 边界预测模型
        "model/C6_918.pt",  # 钙化预测模型
        "model/D8_939.pt",  # 方向预测模型
        "model/S4_930.pt"  # 形状预测模型
    ]

    # 加载四个特征模型
    print("加载四个特征模型...")
    models_4 = []
    for model_path in model_paths_4:
        model = ClassifierLightning4(num_classes=2)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        models_4.append(model)
    print("四个特征模型加载完成！")

    # 加载4特征测试数据
    print("加载4特征测试集...")
    test_dataset_4 = UnlabeledTumorDataset(
        dataset_path="testB/fea",
        transform=transformations
    )
    test_loader_4 = DataLoader(test_dataset_4, batch_size=1, num_workers=0)
    print("4特征测试集加载完成！")

    # 4特征预测
    csv_path_4 = os.path.join(result_dir, 'fea_pre.csv')
    print(f"保存4特征预测结果到 CSV 文件: {csv_path_4}")
    with open(csv_path_4, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['img_name', 'boundary', 'calcification', 'direction', 'shape'])
        for image_tensor, image_name in tqdm(test_loader_4, desc="Predicting 4-features", unit="image"):
            image_tensor = image_tensor.to(device)
            results = [image_name[0]]  # 包括名称和扩展名

            with torch.no_grad():
                for model in models_4:
                    pred = model(image_tensor)
                    prob = torch.sigmoid(pred).squeeze().cpu().numpy()
                    predicted_class_idx = 1 if prob[1] >= 0.5 else 0
                    results.append(predicted_class_idx)

            writer.writerow(results)
    print("4特征预测完成！")