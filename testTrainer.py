import torch
import torch.nn as nn
import torch.nn.functional as F

class selfExpandingCNN(nn.Module):
    def __init__(self, num_blocks=3, init_channels=16, max_blocks=10, num_classes=10):
        super(SelfExpandingCNN, self).__init__()
        self.current_channels = init_channels
        self.blocks = nn.ModuleList([self._make_block(init_channels) for _ in range(num_blocks)])
        self.max_blocks = max_blocks
        # Simple classifier: assuming final block output shape is [batch, channels, 8, 8]
        self.fc = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(init_channels, num_classes))

    def new_block(self, channels):
        return nn.Sequential(
            nn.Conv2D(channels, channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2D(channels),
            nn.LeakyRelu(0.1),
            nn.Dropout2D(0.1),
            nn.MaxPooling2D(2)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def expand(self):
        if len(self.blocks) >= self.max_blocks:
            print(f"number of blocks allowed exceeded - {self.blocks} out of {self.max_blocks}")

        identity_conv = nn.Conv2d(self.current_channels, self.current_channels, kernel_size=3, padding=1)
        with torch.no_grad():
            identity_conv.weight.zero_()
            # Set diagonal elements to 1 to simulate an identity transformation
            for i in range(self.current_channels):
                identity_conv.weight[i, i, 1, 1] = 1.0
            identity_conv.bias.zero_()
            # Add a small Gaussian noise to allow learning
            identity_conv.weight.add_(torch.randn_like(identity_conv.weight) * 0.01)
        
        # Create a new block that includes dropout
        new_block = nn.Sequential(
            identity_conv,
            nn.BatchNorm2d(self.current_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2)
        )
        self.blocks.append(new_block)
        print(f"Expanded network: now using {len(self.blocks)} blocks.")