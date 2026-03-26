import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))
    
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // reduction)
        self.fc2 = nn.Linear(ch // reduction, ch)

    def forward(self, x):
        w = torch.mean(x, dim=-1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)
    
class ECGResNet(nn.Module):
    def __init__(self, in_channels=3, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(32, 64),
            SEBlock(64),
            nn.MaxPool1d(2)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, dilation=2),
            SEBlock(128),
            nn.MaxPool1d(2)
        )

        self.layer3 = ResBlock(128, 128, dilation=4)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)

        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward_with_embedding(self, x):
        # 1. Feature Extraction (Stem + ResBlocks)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 2. Global Pooling to get the "Embedding"
        # x is (batch, 128, 1) -> squeeze to (batch, 128)
        embeds = self.global_pool(x).squeeze(-1)

        # 3. Classification Head
        logits = self.fc(embeds)

        return logits, embeds