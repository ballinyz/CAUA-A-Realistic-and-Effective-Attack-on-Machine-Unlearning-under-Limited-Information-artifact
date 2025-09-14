import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    """Basic ResNet Block"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class getResNet_18(nn.Module):

    def __init__(self, num_classes=10,weight_path=None):
        super(getResNet_18, self).__init__()

        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet Blocks
        self.block1 = self._make_block(64, 64, num_blocks=2, stride=1)  # First ResNet Block
        self.block2 = self._make_block(64, 64, num_blocks=2, stride=1)  # Second ResNet Block
        self.block3 = self._make_block(64, 128, num_blocks=2, stride=2)  # Third ResNet Block with Downsampling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 512)  # 特征映射到更高维度
        self.bn1_fc = nn.BatchNorm1d(512)  # 对于全连接层，假设 fc1 输出通道为128
        self.fc2 = nn.Linear(512, num_classes)  # 输出类别
        # 如果提供了权重路径，则加载权重
        if weight_path is not None:
            self.load_weights(weight_path)


    def _make_block(self, in_channels, out_channels, num_blocks, stride):
        """Helper function to create ResNet blocks"""
        layers = []
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.bn1_fc(x)  # 批归一化

        x = self.fc2(x)
        return x

    def load_weights(self, weight_path):
        try:
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)
            print(f"✅ 权重成功加载自: {weight_path}")
        except Exception as e:
            print(f"❌ 加载权重失败: {e}")
def getVGG_16():
     return 1
def getEfficientNet():
    return 1