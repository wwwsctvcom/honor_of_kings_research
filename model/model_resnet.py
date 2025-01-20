import torch
from torch import nn
from torchvision import models
from model.model_config import ResNetModelConfig


class ResNet(nn.Module):
    """
    examples:
            images = torch.randn((32, 3, 224, 224))
            config = ResNetModelConfig()
            model = AutoGameForImageClassification(config)
            output_action, output_critic = model(images)
            print(output_action.shape)
            print(output_critic.shape)
        """

    def __init__(self, config: ResNetModelConfig):
        super(ResNet, self).__init__()

        # loading model, better use resnet101, IMAGENET1K_V2
        self.model = getattr(models, config.backend_model_name)(weights=models.ResNet101_Weights.DEFAULT)

        # remove classify layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            # (batch_size, channe   ls, height, width) -> (batch_size, channels, output_size, output_size)
            features = features.permute(0, 2, 3, 1)
        return features


class AutoGameForImageClassification(nn.Module):
    def __init__(self, config: ResNetModelConfig):
        super(AutoGameForImageClassification, self).__init__()
        self.config = config
        self.resnet = ResNet(self.config)
        self.classifier_action = nn.Linear(self.config.classify_in_features, self.config.action_dim)
        self.classifier_critic = nn.Linear(self.config.classify_in_features, self.config.critic_dim)

    def forward(self, x):
        shared_representation = self.resnet(x).reshape(-1, self.config.classify_in_features)
        output_action = self.classifier_action(shared_representation)
        output_critic = self.classifier_critic(shared_representation)
        return output_action, output_critic
