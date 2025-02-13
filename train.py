import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import argparse
from sklearn.metrics import r2_score
from models.fusion import Fusion
from evaluate_contribution import evaluate_contribution, write_contribution


def split_train_val_test(data, train_ids, val_ids, test_ids):
    X, Y = data.iloc[:, 1:], data.iloc[:, 0]
    X_train, Y_train = X[X.index.isin(train_ids)], Y[Y.index.isin(train_ids)]
    X_val, Y_val = X[X.index.isin(val_ids)], Y[Y.index.isin(val_ids)]
    X_test, Y_test = X[X.index.isin(test_ids)], Y[Y.index.isin(test_ids)]
    # X : transform to tensor
    X_train = torch.tensor(X_train.values, dtype=torch.float)
    X_val = torch.tensor(X_val.values, dtype=torch.float)
    X_test = torch.tensor(X_test.values, dtype=torch.float)
    # X : reshape
    X_train = X_train.reshape(X_train.shape[0], 1, -1)
    X_val = X_val.reshape(X_val.shape[0], 1, -1)
    X_test = X_test.reshape(X_test.shape[0], 1, -1)

    # Y
    Y_train = torch.tensor(Y_train.values, dtype=torch.float).unsqueeze(1)
    Y_val = torch.tensor(Y_val.values, dtype=torch.float).unsqueeze(1)
    Y_test = torch.tensor(Y_test.values, dtype=torch.float).unsqueeze(1)
    return  X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_data(genotype_path, grm_path, phenotype_path, train_val_ids_path):
    genotype = pd.read_csv(genotype_path, index_col=0).iloc[:100, :100]
    grm = pd.read_csv(grm_path, index_col=0)
    phenotype = pd.read_csv(phenotype_path, index_col=0)
    with open(train_val_ids_path + "train_ids.txt", 'r') as f:
        train_ids = f.read().split(",")
    with open(train_val_ids_path + "val_ids.txt", 'r') as f:
        val_ids = f.read().split(",")
    with open(train_val_ids_path + "test_ids.txt", 'r') as f:
        test_ids = f.read().split(",")
    # merge data
    phenotype_genotype = pd.merge(phenotype, genotype, left_index=True, right_index=True)
    phenotype_grm = pd.merge(phenotype, grm, left_index=True, right_index=True)
    # split data
    snp_train_tensor, phen_train_tensor, snp_val_tensor, phen_val_tensor, snp_test_tensor, phen_test_tensor = split_train_val_test(phenotype_genotype, train_ids,val_ids, test_ids)
    grm_train_tensor, _, grm_val_tensor, _, grm_test_tensor, _ = split_train_val_test(phenotype_grm, train_ids, val_ids, test_ids)
    return snp_train_tensor, grm_train_tensor, phen_train_tensor, \
        snp_val_tensor, grm_val_tensor, phen_val_tensor, \
        snp_test_tensor, grm_test_tensor, phen_test_tensor, \
        torch.tensor(phenotype_genotype.iloc[:, 1:].values, dtype=torch.float), \
        torch.tensor(phenotype_grm.iloc[:, 1:].values, dtype=torch.float)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMNet")
    # dataset parameters
    parser.add_argument("--GRM_epoch", type=int, default=30, help = "The epoch number used to generate and save the Genetic Relatedness Matrix (GRM)")
    parser.add_argument("--genotype_path", type=str, default="data/Genotype.csv", help= "Path to the genotype data file (CSV format)")
    parser.add_argument("--GRM_path", type=str, default="data/", help = "Path to the generated GRM")
    parser.add_argument("--phenotype_path", type=str, default="data/Phenotype.csv", help = "Path to the phenotype data file (CSV format)")
    parser.add_argument("--train_val_ids_path", type=str, default="data/train_val_test/", help = "Path to the directory containing training, validation, and test set indices")

    # training parameters
    parser.add_argument("--epoch", type=int, default=30, help = "Number of iterations")
    parser.add_argument("--p_ve_upper", type=float, default=0.8, help = "The dropout rate for the upper branch of the VE")
    parser.add_argument("--p_ve_lower", type=float, default=0.8, help = "The dropout rate for the lower branch of the VE")
    parser.add_argument("--p_grm", type=float, default=0.8, help = "The dropout rate for the ESN ")
    parser.add_argument("--p_fusion", type=float, default=0.8, help = "The dropout rate for the Fusion module")
    parser.add_argument("--k", type=int, default=2, help = "The number of top-performing models (based on validation performance) to average for evaluation")
    parser.add_argument("--stride", type=int, default=3, help = "The stride for the first convolutional layer in the upper branch of the VE")
    parser.add_argument("--batch_size", type=int, default=128, help = "Number of samples per batch during training")
    parser.add_argument("--lr", type=float, default=0.01, help = "Initial learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help = "L2 regularization strength to prevent overfitting")
    parser.add_argument("--factor", type=float, default=0.5, help = "Factor by which the learning rate is reduced when performance plateaus")
    parser.add_argument("--patience", type=int, default=3, help = "Number of consecutive epochs without improvement before reducing the learning rate")
    parser.add_argument("--monitor", type=str, default='train_loss', help = "The metric to monitor (train_loss or val_loss)")
    args = parser.parse_args()
    print(args)

    # define seed
    seed = args.GRM_epoch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # create DataLoader
    snp_train, grm_train, phen_train, snp_val, grm_val, phen_val, snp_test, grm_test, phen_test, pca_snp, pca_grm  = get_data(args.genotype_path,
                                                                                                                               args.GRM_path + str(args.GRM_epoch) + ".csv",
                                                                                                                               args.phenotype_path,
                                                                                                                               args.train_val_ids_path)
    train_dataloader = DataLoader(TensorDataset(snp_train, grm_train, phen_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(TensorDataset(snp_val, grm_val, phen_val), batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(snp_test, grm_test, phen_test), batch_size=args.batch_size, shuffle=True)

    # define model
    model = Fusion(snp_train.size()[-1], grm_train.size()[-1],
                  p_snp_upper = args.p_ve_upper,
                  p_snp_lower = args.p_ve_lower,
                  p_grm =  args.p_grm,
                  p_fusion = args.p_fusion
                  )
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience)


    k = args.k
    best_models = []
    best_r2_scores = []
    total_snp_contribution_ls = []
    total_grm_contribution_ls = []
    # train
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        num_train = 0
        for batch_train in train_dataloader:
            x1_train, x2_train, y_train = batch_train

            optimizer.zero_grad()
            x1_output_train, x2_output_train, fusion_output_train = model(x1_train, x2_train)
            loss = loss_fn(x1_output_train, y_train) + loss_fn(x2_output_train, y_train) + loss_fn(fusion_output_train, y_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x1_train.shape[0]
            num_train += x1_train.shape[0]
        if args.monitor == 'train_loss':
            scheduler.step(train_loss / num_train)

        # eval
        model.eval()
        val_loss = 0
        num_val = 0
        with torch.no_grad():
            val_true, val_pred = [], []
            for batch_val in val_dataloader:
                x1_val, x2_val, y_val = batch_val

                x1_output_val, x2_output_val, fusion_output_val = model(x1_val, x2_val)
                loss = loss_fn(x1_output_val, y_val) + loss_fn(x2_output_val, y_val) + loss_fn(fusion_output_val, y_val)
                val_loss += loss.item() * x1_val.shape[0]
                num_val += x1_val.shape[0]

                val_true.extend(y_val.numpy())
                val_pred.extend(fusion_output_val.numpy())
            val_true = np.array(val_true).flatten()
            val_pred = np.array(val_pred).flatten()
            val_r2 = r2_score(val_true, val_pred)
        if args.monitor == 'val_loss':
            scheduler.step(val_loss / num_val)

        # save the top k models
        if len(best_r2_scores) < k:
            best_models.append(model)
            best_r2_scores.append(val_r2)
        else:
            min_r2_index = best_r2_scores.index(min(best_r2_scores))
            if val_r2 > best_r2_scores[min_r2_index]:
                best_models[min_r2_index] = model
                best_r2_scores[min_r2_index] = val_r2

        model.eval()
        with torch.no_grad():
            for batch_test in test_dataloader:
                pass

        # evaluate contribution
        total_snp_contribution, total_grm_contribution = evaluate_contribution(model, pca_snp, pca_grm)
        total_snp_contribution_ls.append(total_snp_contribution)
        total_grm_contribution_ls.append(total_grm_contribution)
    # log
    write_contribution(total_snp_contribution_ls, total_grm_contribution_ls)



    # test
    test_r2_scores = []
    for i in range(k):
        model = best_models[i]
        model.eval()
        test_true, test_pred = [], []
        with torch.no_grad():
            for batch_test in test_dataloader:
                x1_test, x2_test, y_test = batch_test
                x1_output_test, x2_output_test, fusion_output_test = model(x1_test, x2_test)
                test_true.extend(y_test.numpy())
                test_pred.extend(fusion_output_test.numpy())
            test_true = np.array(test_true).flatten()
            test_pred = np.array(test_pred).flatten()
            test_r2 = r2_score(test_true, test_pred)
            test_r2_scores.append(test_r2)
    with open('result/test_r2_scores.txt', 'w') as ff:
        for i, score in enumerate(test_r2_scores):
            ff.write(f'Test R2 Score {i + 1}: {score}\n')
        ff.write(f'Mean Test R2 Score: {np.mean(test_r2_scores)}\n')

