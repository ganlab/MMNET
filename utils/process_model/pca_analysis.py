import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.backends.mkl import verbose


def pca_analysis(model, VE, GR):
    model.eval()
    pca_ve_output = []
    pca_esn_output = []
    with torch.no_grad():
        for start in range(0, GR.size(0), 1024):
            end = min(start + 1024, GR.size(0))
            ve_inputs = VE[start:end]
            gr_inputs = GR[start:end]
            # snp
            pca_weight = torch.softmax(model.weight, dim=-1)
            pca_ve_embedding = model.variation_lower(ve_inputs) * pca_weight[0] + model.variation_upper(ve_inputs) * \
                               pca_weight[1]
            # sim
            pca_esn_embedding = model.gr(gr_inputs)
            # BatchNorm1d
            pca_cat = torch.cat((pca_ve_embedding, pca_esn_embedding), dim=1)
            pca_output3 = model.fusion[0](pca_cat)
            # extract
            pca_ve_output.append(pca_output3[:, :1024].detach().cpu().numpy())
            pca_esn_output.append(pca_output3[:, 1024:].detach().cpu().numpy())
        pca_ve_output = np.concatenate(pca_ve_output, axis=0)
        pca_esn_output = np.concatenate(pca_esn_output, axis=0)
        pca_combined_features = np.concatenate([pca_ve_output, pca_esn_output], axis=1)
        n_samples, n_features = pca_combined_features.shape
        n_components = min(n_samples, n_features)
        pca = PCA(n_components=n_components)
        pca.fit(pca_combined_features)
        pca_explained_variance = pca.explained_variance_ratio_
        ve_range = slice(0, 1024)
        esn_range = slice(1024, 2048)
        ve_contributions = np.sum(pca.components_[:, ve_range] ** 2, axis=1)
        esn_contributions = np.sum(pca.components_[:, esn_range] ** 2, axis=1)
        total_contributions = ve_contributions + esn_contributions
        ve_contributions_ratio = ve_contributions / total_contributions
        esn_contributions_ratio = esn_contributions / total_contributions
        pca_ve = np.sum(ve_contributions_ratio * pca_explained_variance)
        pca_esn = np.sum(esn_contributions_ratio * pca_explained_variance)
    return pca_ve, pca_esn

if __name__ == '__main__':
    pass



