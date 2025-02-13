from torch import nn

class Embedding(nn.Module):
    def __init__(self, size):
        super(Embedding, self).__init__()
        self.basenet = nn.Sequential(
            nn.Linear(size, 8192),
        )
    def forward(self, x):
        x = self.basenet(x)
        return x


class Net(nn.Module):
    def __init__(self, size, p):
        super(Net, self).__init__()
        self.embedding = Embedding(size)
        self.network = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(8192, 2048),
            nn.Flatten(),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p),
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.network(x)
        return x

if __name__ == '__main__':
    net = Net(100, p=0.8)
    print(net)

