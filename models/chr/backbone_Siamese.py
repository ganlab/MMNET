from torch import nn
from itertools import accumulate
import torch


def get_windows_linear(windows:int, count:list):
    sum_ls = []
    n = len(count)
    start = 0
    while start < n:
        end = min(start + windows, n)
        sum_ls.append(sum(count[start:end]))
        start = end
    base_output_dim = 8192 // len(sum_ls)
    remainder = 8192 % len(sum_ls)
    output_dim_ls = []
    for i in range(len(sum_ls)):
        output_dim_ls.append(base_output_dim + (1 if i < remainder else 0))
    return sum_ls, output_dim_ls


class Embedding(nn.Module):
    def __init__(self, windows:int, count:list):
        super(Embedding, self).__init__()
        self.input_dim, self.output_dim = get_windows_linear(windows, count)
        self.basenet = [nn.Linear(s, o) for s, o in zip(self.input_dim, self.output_dim)]

    def forward(self, x):
        cumulative_sum = [0] + list(accumulate(self.input_dim))
        temp = []
        for i in range(len(self.basenet)):
           temp.append(self.basenet[i](x[:, :, cumulative_sum[i]:cumulative_sum[i+1]]))
        x = torch.cat(temp, dim=-1)
        return x


class ChrNet(nn.Module):
    def __init__(self, p, windows:int, count:list):
        super(ChrNet, self).__init__()
        self.embedding = Embedding(windows, count)
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
    pass
