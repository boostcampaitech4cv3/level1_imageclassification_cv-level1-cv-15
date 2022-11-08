import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

from efficientnet_pytorch import EfficientNet as EffNet
from pytorch_pretrained_vit import ViT

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None: 
            nn.init.constant_(m.bias, 0.0)

class VisionTransformer(nn.Module):
    def __init__(self, triplet, num_classes=18, dropout_rate=0.1,  selected="L_16_imagenet1k", pretrained=True) -> None:
        super().__init__()
        self.selected_model = "B_16_imagenet1k" if "B_16" in selected else selected
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.triplet = triplet
        self.model = ViT(self.selected_model, pretrained=self.pretrained, dropout_rate=self.dropout_rate,
         num_classes=self.num_classes, image_size=224, patches=16)

        self.model.fc = nn.Identity()
        self.classifier = nn.Linear(in_features = 1024, out_features = num_classes, bias = True)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x):

        if self.triplet:
            feat = self.model(x)
            score = self.classifier(feat)
            return feat, score
        else:
            x = self.model(x)
            x = self.classifier(x)
            return x


class EfficientNet_b0(nn.Module):
    def __init__(
        self,
        num_classes,
        triplet,
        pretrained = True,
    ):
        super().__init__()

        if pretrained:
            self.model = EffNet.from_pretrained("efficientnet-b0")
        else:
            self.model = EffNet.from_name("efficientnet-b0")

        self.fc = nn.Linear(1000, num_classes)
        self.triplet = triplet

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        # x shape: batch_size, 3, 128, 96
        if self.triplet:
            feat = self.model(x)
            score = self.fc(feat)
            
            return feat, score
        else:
            x = self.model(x)
            x = self.fc(x)
            # x shape: batch_size, num_classes
            return x

class EfficientNet_b4(nn.Module):
    def __init__(
        self,
        num_classes,
        triplet,
        pretrained = True,
    ):
        super().__init__()

        if pretrained:
            self.model = EffNet.from_pretrained("efficientnet-b4")
        else:
            self.model = EffNet.from_name("efficientnet-b4")

        self.fc = nn.Linear(1000, num_classes)
        self.triplet = triplet

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        # x shape: batch_size, 3, 128, 96
        if self.triplet:
            feat = self.model(x)
            score = self.fc(feat)
            
            return feat, score
        else:
            x = self.model(x)
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