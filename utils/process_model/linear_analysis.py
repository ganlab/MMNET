import torch

def linear_analysis(model):
    model.eval()
    with torch.no_grad():
        linear_layer = model.fusion[2]
        weights = linear_layer.weight.data
        weights_ve = weights[:, :1024]
        weights_esn = weights[:, 1024:]
        contribution_ve = torch.norm(weights_ve, p=2, dim=1)  # 每个输出节点对 scale1 的贡献
        contribution_esn = torch.norm(weights_esn, p=2, dim=1)
        total_contribution_ve = contribution_ve.sum().item()
        total_contribution_esn = contribution_esn.sum().item()
        linear_ve = total_contribution_ve / (total_contribution_ve + total_contribution_esn)
        linear_esn = total_contribution_esn / (total_contribution_ve + total_contribution_esn)
    return linear_ve, linear_esn


if __name__ == '__main__':
    pass