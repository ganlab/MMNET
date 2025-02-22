import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.backbone_Siamese import Net
import numpy as np
import random
import argparse

def get_data(genotype_path, phenotype_path, train_val_ids_path):
    genotype = pd.read_csv(genotype_path, index_col=0)
    phenotype = pd.read_csv(phenotype_path, index_col=0)
    with open(train_val_ids_path + "train_ids.txt", 'r') as f:
        train_ids = f.read().split(",")
    with open(train_val_ids_path + "val_ids.txt", 'r') as f:
        val_ids = f.read().split(",")
    data = pd.merge(phenotype, genotype, left_index=True, right_index=True)
    data_train, data_val = data[data.index.isin(train_ids)], data[data.index.isin(val_ids)]
    snp_train, phen_train = data_train.iloc[:, 1:], data_train.iloc[:, 0]
    snp_val, phen_val = data_val.iloc[:, 1:], data_val.iloc[:, 0]

    all_sample = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float)
    snp_train = torch.tensor(snp_train.values, dtype=torch.float)
    snp_val = torch.tensor(snp_val.values, dtype=torch.float)
    snp_train = snp_train.reshape(snp_train.shape[0], 1, -1)
    snp_val = snp_val.reshape(snp_val.shape[0], 1, -1)
    phen_train = torch.tensor(phen_train.values, dtype=torch.float).unsqueeze(1)
    phen_val = torch.tensor(phen_val.values, dtype=torch.float).unsqueeze(1)
    return all_sample, snp_train, phen_train, snp_val, phen_val

def extract_embedding(best_model, genotype_path, phenotype_path):
    # extract feature inputs
    genotype = pd.read_csv(genotype_path, index_col=0)
    phenotype = pd.read_csv(phenotype_path, index_col=0)
    data_merge = pd.merge(genotype, phenotype, left_index=True, right_index=True)
    feature = data_merge.iloc[:, :-1]
    index = feature.index
    feature = torch.tensor(feature.values, dtype=torch.float)

    # extract embedding
    best_model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, feature.size()[0], 1024):
            end = min(start + 1024, feature.size()[0])
            embeddings.append(best_model.embedding(feature[start:end]).numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return pd.DataFrame(embeddings, index=index)

def generate_GRM(df, GRM_path, epoch):
    df_tensor = torch.tensor(df.values, dtype=torch.float)
    distance_matrix = torch.cdist(df_tensor, df_tensor, p=2)
    max_distance = torch.max(distance_matrix).item()
    GRM = 1  - distance_matrix / max_distance
    GRM_df = pd.DataFrame(GRM.numpy(), index=df.index, columns=df.index)
    GRM_df.index.name = 'FID'
    GRM_df.astype("float16")
    GRM_df.to_csv(GRM_path + str(epoch) + ".csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMNet")
    # dataset parameters
    parser.add_argument("--genotype_path", type=str, default="data/Genotype.csv", help = "Path to the genotype data file (CSV format)")
    parser.add_argument("--phenotype_path", type=str, default="data/Phenotype.csv", help = "Path to the phenotype data file (CSV format)")
    parser.add_argument("--train_val_ids_path", type=str, default="data/train_val_test/", help = "Path to the directory containing training, validation, and test set indices")

    # training parameters
    parser.add_argument("--epoch", type=int, default=30, help = "Number of iterations")
    parser.add_argument("--p", type=float, default=0.8, help = "Dropout rate")
    parser.add_argument("--batch_size", type=int, default=128, help = "Number of samples per batch during training")
    parser.add_argument("--lr", type=float, default=0.01, help = "Initial learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help = "L2 regularization strength to prevent overfitting")
    parser.add_argument("--factor", type=float, default=0.5, help = "Factor by which the learning rate is reduced when performance plateaus")
    parser.add_argument("--patience", type=int, default=3, help = "Number of consecutive epochs without improvement before reducing the learning rate")

    # path to save GRM
    parser.add_argument("--GRM_path", type=str, default="data/", help = "Path to save the generated GRM")
    args = parser.parse_args()
    print(args)
    # define seed
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # create DataLoader
    all_sample_tensor, snp_train_tensor, phen_train_tensor, snp_val_tensor, phen_val_tensor = get_data(args.genotype_path, args.phenotype_path, args.train_val_ids_path)
    train_dataloader = DataLoader(TensorDataset(snp_train_tensor, phen_train_tensor), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(TensorDataset(snp_val_tensor, phen_val_tensor), batch_size=args.batch_size, shuffle=False)

    # define model
    model = Net(snp_train_tensor.size()[-1], args.p)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience)

    # model train and eval
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        n = 0
        for batch_train in train_dataloader:
            x_train, y_train = batch_train
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_train.shape[0]
            n += x_train.shape[0]
        scheduler.step(train_loss / n)


        model.eval()
        with torch.no_grad():
            for batch_val in val_dataloader:
                x_val, y_val = batch_val
                outputs_val = model(x_val)
                loss_val = loss_fn(outputs_val, y_val)

    # extract embedding
    embedding_df = extract_embedding(model, args.genotype_path, args.phenotype_path)

    # generate_GRM
    generate_GRM(embedding_df, args.GRM_path, args.epoch)



