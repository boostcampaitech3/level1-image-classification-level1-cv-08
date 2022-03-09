import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


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
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x
    
class MyEfficientNet(nn.Module) :
    '''
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 18개의 Class를 예측하는 형태의 Model입니다.
    '''
    def __init__(self, num_classes) :
        super(MyEfficientNet, self).__init__()
        self.EFF = EfficientNet.from_pretrained('efficientnet-b4', in_channels=3, num_classes=18)
    
    def forward(self, x) :
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x
    

class VanillaEfficientNet(nn.Module):
    def __init__(self, num_classes: int, freeze: bool = False):
        super(VanillaEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b3")
        if freeze:
            self._freeze()
        self.batchnorm = nn.BatchNorm1d(num_features=1000)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.relu(x)
        output = self.linear(x)
        return output

    def _freeze(self):
        for param in self.efficientnet.parameters(): # backbone 모델의 파라미터가 학습되지 않도록 통제
            param.requires_grad = False