import torch
from torch import nn
from torchvision import models


class ResNetEmbeddings(nn.Module):
    """
    examples:
            transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # 加载示例图像
            url = "screenshot.jpg"  # 替换为你的图像 URL
            # response = requests.get(url)
            img = Image.open(url).convert('RGB')
            img = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

            # 创建模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ImageEmbedding('resnet101').to(device)
            img = img.to(device)

            # 提取图像嵌入
            embedding = model(img)

            # 打印输入和输出的维度
            print(f"Input shape: {img.shape}")  # (batch_size, 3, 224, 224)
            print(f"Output shape: {embedding.shape}")  # (batch_size, 2048)
        """

    def __init__(self, backend_model_name="resnet101"):
        super(ResNetEmbeddings, self).__init__()

        # loading model, better use resnet101
        # self.model = getattr(models, backend_model_name)(pretrained=True)
        self.model = getattr(models, backend_model_name)(weights=models.ResNet101_Weights.DEFAULT)  # IMAGENET1K_V2

        # remove adaptive pool layer/classify layer
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        # (batch_size, channels, height, width) -> (batch_size, channels, output_size, output_size)
        self.aap = nn.AdaptiveAvgPool2d(output_size=(6, 6))

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            features = self.aap(features)
            features = features.permute(0, 2, 3, 1)
        return features
