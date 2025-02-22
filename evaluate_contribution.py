import torch
from sklearn.decomposition import PCA
import numpy as np

def get_combined_embedding(model, snp, genetic_relatedness):
    snp = snp.reshape(snp.shape[0], 1, -1)
    genetic_relatedness = genetic_relatedness.reshape(genetic_relatedness.shape[0], 1, -1)

    model.eval()
    snp_output = []
    genetic_relatedness_output = []
    with torch.no_grad():
        for start in range(0, snp.size()[0], 1024):
            end = min(start + 1024, snp.size()[0])
            snp_inputs = snp[start:end]
            genetic_relatedness_inputs = genetic_relatedness[start:end]

            # snp embedding
            weight = torch.softmax(model.weight, dim=-1)
            snp_embedding = model.ve_upper(snp_inputs) * weight[0] + model.ve_lower(snp_inputs) * weight[1]

            # genetic_relatedness embedding
            genetic_relatedness_embedding = model.esn(genetic_relatedness_inputs)

            # cat embedding
            cat = torch.cat([snp_embedding, genetic_relatedness_embedding], dim=1)
            embedding = model.fusion[0](cat)
            snp_output.append(embedding[:, :1024].numpy())
            genetic_relatedness_output.append(embedding[:, 1024:].numpy())
        snp_output = np.concatenate(snp_output, axis=0)
        genetic_relatedness_output = np.concatenate(genetic_relatedness_output, axis=0)
        combined_output = np.concatenate([snp_output, genetic_relatedness_output], axis=1)
        return combined_output


def evaluate_contribution(model, snp, genetic_relatedness):
    combined_embedding = get_combined_embedding(model, snp, genetic_relatedness)
    n_samples, n_features = combined_embedding.shape
    pca = PCA(n_components=min(n_samples, n_features))
    pca.fit(combined_embedding)
    pca_explained_variance = pca.explained_variance_ratio_
    snp_range = slice(0, 1024)
    genetic_relatedness_range = slice(1024, 2048)
    snp_contributions = np.sum(pca.components_[:, snp_range] ** 2, axis=1)
    genetic_relatedness_contributions = np.sum(pca.components_[:, genetic_relatedness_range] ** 2, axis=1)
    total_contributions = snp_contributions + genetic_relatedness_contributions
    snp_contributions_ratio = snp_contributions / total_contributions
    genetic_relatedness_contributions_ratio = genetic_relatedness_contributions / total_contributions
    total_snp_contribution = np.sum(snp_contributions_ratio * pca_explained_variance)
    total_genetic_relatedness_contribution = np.sum(genetic_relatedness_contributions_ratio * pca_explained_variance)
    return total_snp_contribution, total_genetic_relatedness_contribution

def write_contribution(snp, genetic_relatedness):
    with open(f"result/ve_contribution.txt", 'w') as f:
        f.write(",".join([str(ss) for ss in snp]))
    with open(f"result/esn_contribution.txt", 'w') as f:
        f.write(",".join([str(ss) for ss in genetic_relatedness]))

if __name__ == '__main__':
    pass