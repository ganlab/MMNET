from sklearn.model_selection import train_test_split
import pandas as pd
import random

# Load the gene data
gene = pd.read_csv("data/Genotype.csv", index_col=0)

indices = list(gene.index)
train_ids, test_ids = train_test_split(indices, train_size=0.7, random_state=0)

random.seed(42)
num_test = len(test_ids)
num_val = int(num_test * 1 / 3)
random.shuffle(test_ids)
val_ids = test_ids[:num_val]
test_ids = test_ids[num_val:]


# Save the indices to respective files
with open("data/train_val_test/train_ids.txt", "w") as f:
    f.write(",".join(map(str, train_ids)))

with open("data/train_val_test/test_ids.txt", "w") as f:
    f.write(",".join(map(str, test_ids)))

with open("data/train_val_test/val_ids.txt", "w") as f:
    f.write(",".join(map(str, val_ids)))

