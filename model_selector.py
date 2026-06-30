# model_selector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet
from torchvision.models import resnet18, efficientnet_b0
from torchvision.models._utils import IntermediateLayerGetter

class ClassifierWrapper(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ClassifierWrapper, self).__init__()
        self.base = base_model
        self.head = nn.Linear(self.base.fc.in_features, num_classes)
        self.base.fc = self.head

    def forward(self, x):
        return self.base(x)

class UNetResNet18(nn.Module):
    def __init__(self, out_channels=6, dropout=0.3):
        super(UNetResNet18, self).__init__()

        # Pretrained ResNet18 encoder
        backbone = resnet18(pretrained=True)
        self.encoder = IntermediateLayerGetter(
            backbone,
            return_layers={
                "relu": "layer0",
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4"
            }
        )
        self.dropout = dropout

        # Decoder (upsampling) with dropout for regularization
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        x4 = feats["layer4"]
        x3 = feats["layer3"]
        x2 = feats["layer2"]
        x1 = feats["layer1"]
        x0 = feats["layer0"]

        u4 = self.up4(x4)
        c4 = self.conv4(torch.cat([u4, x3], dim=1))
        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([u3, x2], dim=1))
        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, x1], dim=1))
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, x0], dim=1))

        # Upsample final output to match input size
        return F.interpolate(self.out(c1), size=(256, 256), mode='bilinear', align_corners=False)




#UNET PLUS PLUS
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------
# Basic VGG-style block: two conv→BN→ReLU layers
# ------------------------------------------------------------------------
class VGGBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,    mid_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch,   out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# ------------------------------------------------------------------------
# U‑Net++ (Nested U‑Net) implementation
# ------------------------------------------------------------------------
class NestedUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=6, deep_supervision=False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Level 0
        self.conv0_0 = VGGBlock(input_channels,  nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # Nested skip convs
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1],       nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2],       nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3],       nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4],       nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2 + nb_filter[1],     nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2 + nb_filter[2],     nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2 + nb_filter[3],     nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3 + nb_filter[1],     nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3 + nb_filter[2],     nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4 + nb_filter[1],     nb_filter[0], nb_filter[0])

        # Output heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, 1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, 1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, 1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, 1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, 1)

    def forward(self, x):
        # Encoder path
        x0_0 = self.conv0_0(x)               # → [B,32,H,W]
        x1_0 = self.conv1_0(self.pool(x0_0)) # → [B,64,H/2,W/2]
        x2_0 = self.conv2_0(self.pool(x1_0)) # → [B,128,H/4,W/4]
        x3_0 = self.conv3_0(self.pool(x2_0)) # → [B,256,H/8,W/8]
        x4_0 = self.conv4_0(self.pool(x3_0)) # → [B,512,H/16,W/16]

        # Nested convs level 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)],  1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)],  1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)],  1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)],  1))

        # Nested convs level 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))  # now input ch=128+128+256=512

        # Nested convs level 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # Nested conv level 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Final output
        if self.deep_supervision:
            return [
                self.final1(x0_1),
                self.final2(x0_2),
                self.final3(x0_3),
                self.final4(x0_4),
            ]
        else:
            return self.final(x0_4)



def get_model(model_type, in_channels=3, out_channels=6, num_classes=6, dropout=0.2, task="segmentation"):
    if task == "segmentation":
        if model_type.lower() == "unet":
            return UNet(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
        elif model_type.lower() == "unetplusplus":
            return NestedUNet(num_classes=out_channels, input_channels=in_channels, deep_supervision=False)
        elif model_type.lower() == "unet_resnet18":
            return UNetResNet18(out_channels=out_channels, dropout=dropout)
        else:
            raise ValueError("Unsupported segmentation model: " + model_type)

    elif task == "classification":
        if model_type.lower() == "resnet18":
            model = resnet18(pretrained=True)
            return ClassifierWrapper(model, num_classes)
        elif model_type.lower() == "efficientnet":
            model = efficientnet_b0(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model
        else:
            raise ValueError("Unsupported classification model: " + model_type)

    else:
        raise ValueError("Unsupported task type. Use 'segmentation' or 'classification'.")