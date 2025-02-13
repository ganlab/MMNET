### Parameter to train MMNet

- The parameter adjustments comprehensively consider regression evaluation metrics such as R², PCC, MAE, and MSE. Users can fine-tune them as needed based on their specific requirements.

  **linkage disequilibrium (ld) = 0.5**

  | Phentype              | learning rate | stride | batch_size | monitor    |
  | --------------------- | ------------- | ------ | ---------- | ---------- |
  | culm length           | 0.01          | 3      | 128        | train_loss |
  | grain length          | 0.01          | 3      | 128        | val_loss   |
  | grain width           | 0.01          | 3      | 128        | val_loss   |
  | grain yield           | 0.01          | 3      | 160        | val_loss   |
  | heading date          | 0.01          | 3      | 128        | train_loss |
  | leaf angle            | 0.01          | 3      | 128        | val_loss   |
  | leaf length           | 0.01          | 3      | 128        | val_loss   |
  | leaf width            | 0.01          | 3      | 128        | val_loss   |
  | panicle length        | 0.01          | 3      | 128        | val_loss   |
  | panicle number        | 0.01          | 3      | 64         | val_loss   |
  | plant height          | 0.01          | 3      | 128        | train_loss |
  | grain protein content | 0.01          | 3      | 128        | val_loss   |

  **linkage disequilibrium (ld) = 0.4**

  | Phentype              | learning rate | stride | batch_size | monitor    |
  | --------------------- | ------------- | ------ | ---------- | ---------- |
  | culm length           | 0.01          | 1      | 128        | train_loss |
  | grain length          | 0.01          | 1      | 128        | val_loss   |
  | grain                 | 0.01          | 1      | 128        | val_loss   |
  | grain yield           | 0.01          | 1      | 128        | val_loss   |
  | heading date          | 0.01          | 1      | 128        | train_loss |
  | leaf angle            | 0.01          | 1      | 128        | val_loss   |
  | leaf length           | 0.01          | 1      | 128        | val_loss   |
  | leaf width            | 0.01          | 1      | 128        | val_loss   |
  | panicle length        | 0.01          | 1      | 128        | val_loss   |
  | panicle number        | 0.01          | 1      | 128        | val_loss   |
  | plant height          | 0.01          | 1      | 128        | train_loss |
  | grain protein content | 0.01          | 1      | 32         | val_loss   |

  **linkage disequilibrium (ld) = 0.3**

  | Phentype              | learning rate | stride | batch_size | monitor    |
  | --------------------- | ------------- | ------ | ---------- | ---------- |
  | culm length           | 0.01          | 1      | 128        | train_loss |
  | grain length          | 0.01          | 1      | 64         | val_loss   |
  | grain                 | 0.01          | 1      | 128        | val_loss   |
  | grain yield           | 0.01          | 1      | 128        | val_loss   |
  | heading date          | 0.01          | 1      | 128        | train_loss |
  | leaf angle            | 0.01          | 1      | 128        | val_loss   |
  | leaf length           | 0.01          | 1      | 128        | val_loss   |
  | leaf width            | 0.01          | 1      | 128        | val_loss   |
  | panicle length        | 0.01          | 1      | 128        | val_loss   |
  | panicle number        | 0.01          | 1      | 128        | val_loss   |
  | plant height          | 0.01          | 1      | 128        | train_loss |
  | grain protein content | 0.01          | 1      | 32         | val_loss   |

  **linkage disequilibrium (ld) = 0.2**

  | Phentype              | learning rate | stride | batch_size | monitor    |
  | --------------------- | ------------- | ------ | ---------- | ---------- |
  | culm length           | 0.01          | 1      | 128        | train_loss |
  | grain length          | 0.01          | 1      | 64         | val_loss   |
  | grain                 | 0.01          | 1      | 128        | val_loss   |
  | grain yield           | 0.01          | 1      | 128        | val_loss   |
  | heading date          | 0.01          | 1      | 128        | train_loss |
  | leaf angle            | 0.01          | 1      | 128        | val_loss   |
  | leaf length           | 0.01          | 1      | 128        | val_loss   |
  | leaf width            | 0.01          | 1      | 128        | val_loss   |
  | panicle length        | 0.01          | 1      | 64         | val_loss   |
  | panicle number        | 0.01          | 1      | 32         | val_loss   |
  | plant height          | 0.01          | 1      | 128        | train_loss |
  | grain protein content | 0.01          | 1      | 128        | val_loss   |

  **linkage disequilibrium (ld) = 0.1**

  | Phentype              | learning rate | stride | batch_size | monitor    |
  | --------------------- | ------------- | ------ | ---------- | ---------- |
  | culm length           | 0.01          | 1      | 128        | train_loss |
  | grain length          | 0.01          | 1      | 64         | val_loss   |
  | grain width           | 0.01          | 1      | 32         | val_loss   |
  | grain yield           | 0.01          | 1      | 128        | val_loss   |
  | heading date          | 0.01          | 1      | 128        | train_loss |
  | leaf angle            | 0.01          | 1      | 128        | val_loss   |
  | leaf length           | 0.01          | 1      | 128        | val_loss   |
  | leaf width            | 0.01          | 1      | 128        | val_loss   |
  | panicle length        | 0.01          | 1      | 64         | val_loss   |
  | panicle number        | 0.01          | 1      | 32         | val_loss   |
  | plant height          | 0.01          | 1      | 10         | train_loss |
  | grain protein content | 0.01          | 1      | 32         | val_loss   |

**Tips:** It is necessary to analyze the loss variations of the training and validation sets during the training process and adjust the dropout rates of each module accordingly.