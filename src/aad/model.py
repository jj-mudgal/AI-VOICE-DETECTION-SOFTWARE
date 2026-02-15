import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioDetector(nn.Module):
    def __init__(self):
        super(AudioDetector, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
        self.fc = nn.Linear(128, 2)  # 2 outputs → Real vs Synthetic

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.mean(x, dim=-1)  # Global Average Pooling
        return self.fc(x)
