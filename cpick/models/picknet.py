"""Depth-only pick policy: output-stride-8 ResNet34 encoder, U-Net-style decoder to 128x128 logits,
value head at the bottleneck. Torch lives here; cpick/sim and cpick/envs never import it."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

GRID = 128


def dilate_resnet_(net):
    """In place: torchvision ResNet (BasicBlock) -> output stride 8. layer3 dilation 2, layer4 dilation 4.
    First block of each layer keeps the previous dilation in conv1 (as torchvision does for Bottleneck)."""
    prev = 1
    for layer, d in ((net.layer3, 2), (net.layer4, 4)):
        for k, block in enumerate(layer):
            d1 = prev if k == 0 else d
            block.conv1.stride = (1, 1)
            block.conv1.dilation, block.conv1.padding = (d1, d1), (d1, d1)
            block.conv2.dilation, block.conv2.padding = (d, d), (d, d)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        prev = d
    return net


def resnet34_8s(pretrained=True):
    w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    return dilate_resnet_(torchvision.models.resnet34(weights=w))


class PickNet(nn.Module):
    def __init__(self, pretrained=True, width=64):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)                           # per-image depth normalisation
        r = resnet34_8s(pretrained)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # 64  @ 128
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        lat = lambda c: nn.Sequential(nn.Conv2d(c, width, 1), nn.ReLU(inplace=True))
        self.lat0, self.lat1, self.lat2, self.lat3, self.lat4 = lat(64), lat(64), lat(128), lat(256), lat(512)
        conv = lambda: nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True))
        self.fuse32, self.fuse64, self.fuse128 = conv(), conv(), conv()
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)                            # initial policy = uniform in mask
        nn.init.zeros_(self.head.bias)
        self.value = nn.Sequential(nn.Conv2d(512, 16, 1), nn.ReLU(inplace=True), nn.Flatten(),
                                   nn.Linear(16 * 32 * 32, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))

    def forward(self, depth):
        """depth (B, 256, 256) metric z-depth -> logits (B, 128, 128), value (B,)."""
        x = self.norm(depth.unsqueeze(1)).repeat(1, 3, 1, 1)
        s0 = self.stem(x)                     # 64  @ 128
        s1 = self.layer1(self.pool(s0))       # 64  @ 64
        s2 = self.layer2(s1)                  # 128 @ 32
        s3 = self.layer3(s2)                  # 256 @ 32 (dilated)
        s4 = self.layer4(s3)                  # 512 @ 32 (dilated)
        y = self.fuse32(self.lat4(s4) + self.lat3(s3) + self.lat2(s2))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.fuse64(y + self.lat1(s1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.fuse128(y + self.lat0(s0))
        return self.head(y).squeeze(1), torch.sigmoid(self.value(s4)).squeeze(1)


def masked_dist(logits, mask):
    """Categorical over 128*128 cells, zero probability outside the mask. Kept in float32
    (a -1e9 fill would overflow to -inf under fp16 and turn the entropy into NaN)."""
    flat = logits.float().flatten(1).masked_fill(~mask.flatten(1).bool(), -1e9)
    return torch.distributions.Categorical(logits=flat)