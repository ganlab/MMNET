import torch
from genetic_relatedness_embedding import *
from ve_embedding import *

class NormalFusion(nn.Module):
    def __init__(self, snp_size, genetic_relatedness_size, p_snp_upper, p_snp_lower, p_genetic_relatedness, p_fusion, stride=3):
        super(NormalFusion, self).__init__()
        self.ve_upper = VEUpperBranch(p_snp_upper, stride=stride)
        self.ve_lower = VELowerBranch(snp_size, p_snp_lower)
        self.weight = nn.Parameter(torch.Tensor([0.5, 0.5]))

        self.esn = GeneticRelatednessEmbedding(genetic_relatedness_size, p_genetic_relatedness)

        self.snp_output = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p_fusion),
            nn.Linear(256, 1),
        )

        self.sim_output = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p_fusion),
            nn.Linear(256, 1),
        )

        self.fusion = nn.Sequential(
            nn.BatchNorm1d(1024 * 2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(1024 * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p_fusion),
            nn.Linear(512, 1),
        )

    def forward(self, snp, genetic_relatedness):
        snp_embedding_upper = self.ve_upper(snp)
        snp_embedding_lower = self.ve_lower(snp)
        weight = torch.softmax(self.weight, dim=-1)
        snp_embedding = weight[0] * snp_embedding_upper + weight[1] * snp_embedding_lower
        genetic_relatedness_embedding = self.esn(genetic_relatedness)
        snp_output = self.snp_output(snp_embedding)
        sim_output = self.sim_output(genetic_relatedness_embedding)
        fusion_output = self.fusion(torch.cat([snp_embedding, genetic_relatedness_embedding], dim=-1))
        return snp_output, sim_output, fusion_output

if __name__ == '__main__':
    pass

