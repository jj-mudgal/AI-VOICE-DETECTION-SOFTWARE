import torch
import torch.nn as nn
import torchaudio.transforms as T


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

        self.se = SEBlock(out_ch)

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)
        out += identity
        return self.relu(out)


class AudioDetector(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()

        # 🔥 moved mel inside model
        self.mel = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )

        self.to_db = T.AmplitudeToDB()

        self.backbone = nn.Sequential(
            ResidualBlock(1, 64),
            nn.MaxPool2d(2),

            ResidualBlock(64, 128),
            nn.MaxPool2d(2),

            ResidualBlock(128, 256),
            nn.MaxPool2d(2),

            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, waveform):
        x = self.mel(waveform)
        x = self.to_db(x)

        x = x.unsqueeze(1)  # (B, 1, mel, time)

        x = self.backbone(x)
        return self.head(x)
