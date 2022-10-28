import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

from efficientnet_pytorch import EfficientNet as EffNet

class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained = True,
    ):
        super().__init__()

        if pretrained:
            self.model = EffNet.from_pretrained("efficientnet-b0")
        else:
            self.model = EffNet.from_name("efficientnet-b0")

        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        # x shape: batch_size, 3, 128, 96

        x = self.model(x)
        # x shape: batch_size, 1000

        x = self.fc(x)
        # x shape: batch_size, num_classes

        return x




class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template

# Example: ResNet34
# num_classes = 18

class ResNet34(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(in_features = 512, out_features = num_classes, bias = True)
        nn.init.xavier_uniform(self.model.fc.weight)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

class ResNet152(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = torchvision.models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(in_features = 2048, out_features = num_classes, bias = True)
        nn.init.xavier_uniform(self.model.fc.weight)
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x
