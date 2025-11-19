import torch
import torch.nn as nn
import torchvision.models as models
import config

class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, n_div=8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div
        
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(nt, c, h, w)

class TSMRnet50(nn.Module):
    def __init__(self, num_classes, num_segments):
        super(TSMRnet50, self).__init__()
        self.num_segments = num_segments
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = self._make_tsm_layer(resnet.layer1)
        self.layer2 = self._make_tsm_layer(resnet.layer2)
        self.layer3 = self._make_tsm_layer(resnet.layer3)
        self.layer4 = self._make_tsm_layer(resnet.layer4)
        
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def _make_tsm_layer(self, layer):
        blocks = list(layer.children())
        for block in blocks:
            if hasattr(block, 'conv1'):
                block.conv1 = nn.Sequential(
                    TemporalShift(self.num_segments),
                    block.conv1
                )
        return nn.Sequential(*blocks)

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.view(b * t, c, h, w)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = x.view(b, t, -1)
        x = x.mean(dim=1)
        
        return x