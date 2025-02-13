from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        kernel_size = 3
        stride = 2
        self.inner = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,stride=stride),
            nn.BatchNorm1d(out_channels),
        )

        self.outer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,kernel_size=1, stride=4),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        x = self.inner(x) + self.outer(x)
        return x

class VEUpperBranch(nn.Module):
    def __init__(self, p, stride=3):
        super(VEUpperBranch, self).__init__()
        self.upper = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=32, padding=1, stride=stride),
            ResidualBlock(8, 16),
            nn.MaxPool1d(2),
            ResidualBlock(16, 64),
            nn.Flatten(),

            # Modify the number of input neurons to accommodate the stride parameter and the dimensions of your input data.
            nn.Linear(243 * 64, 4096),

            nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(4096, 1024),
        )
    def forward(self, x):
        x = self.upper(x)
        return x


class VELowerBranch(nn.Module):
    def __init__(self, snp_size, p):
        super(VELowerBranch, self).__init__()
        self.lower = nn.Sequential(
            nn.Linear(snp_size, 4096),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(4096, 1024),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.lower(x)
        return x

if __name__ == "__main__":
    pass