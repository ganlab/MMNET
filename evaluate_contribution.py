import torch
from sklearn.decomposition import PCA
from models.mmnet import MMNet
import numpy as np

def get_combined_embedding(model:MMNet, snp, grm):
    snp = snp.reshape(snp.shape[0], 1, -1)
    grm = grm.reshape(grm.shape[0], 1, -1)

    model.eval()
    snp_output = []
    grm_output = []
    with torch.no_grad():
        for start in range(0, snp.size()[0], 1024):
            end = min(start + 1024, snp.size()[0])
            snp_inputs = snp[start:end]
            grm_inputs = grm[start:end]

            # snp embedding
            weight = torch.softmax(model.weight, dim=-1)
            snp_embedding = model.ve_upper(snp_inputs) * weight[0] + model.ve_lower(snp_inputs) * weight[1]

            # grm embedding
            grm_embedding = model.esn(grm_inputs)

            # cat embedding
            cat = torch.cat([snp_embedding, grm_embedding], dim=1)
            embedding = model.fusion[0](cat)
            snp_output.append(embedding[:, :1024].numpy())
            grm_output.append(embedding[:, 1024:].numpy())
        snp_output = np.concatenate(snp_output, axis=0)
        grm_output = np.concatenate(grm_output, axis=0)
        combined_output = np.concatenate([snp_output, grm_output], axis=1)
        return combined_output


def evaluate_contribution(model:MMNet, snp, grm):
    combined_embedding = get_combined_embedding(model, snp, grm)
    n_samples, n_features = combined_embedding.shape
    pca = PCA(n_components=min(n_samples, n_features))
    pca.fit(combined_embedding)
    pca_explained_variance = pca.explained_variance_ratio_
    snp_range = slice(0, 1024)
    grm_range = slice(1024, 2048)
    snp_contributions = np.sum(pca.components_[:, snp_range] ** 2, axis=1)
    grm_contributions = np.sum(pca.components_[:, grm_range] ** 2, axis=1)
    total_contributions = snp_contributions + grm_contributions
    snp_contributions_ratio = snp_contributions / total_contributions
    grm_contributions_ratio = grm_contributions / total_contributions
    total_snp_contribution = np.sum(snp_contributions_ratio * pca_explained_variance)
    total_grm_contribution = np.sum(grm_contributions_ratio * pca_explained_variance)
    return total_snp_contribution, total_grm_contribution

def write_contribution(snp, grm):
    with open(f"result/ve_contribution.txt", 'w') as f:
        f.write(",".join([str(ss) for ss in snp]))
    with open(f"result/esn_contribution.txt", 'w') as f:
        f.write(",".join([str(ss) for ss in grm]))

if __name__ == '__main__':
    pass