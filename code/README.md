# Code

This repository contains source code for our paper. The code includes data processing, model building, and visualization of results.

> [!IMPORTANT]
> We use jupyter notebooks for visualization purposes only. Please, keep your code in Python scripts.

## Installation

Clone the repo:
```bash
git clone https://github.com/intsystems/2026-Project-203.git
cd 2026-Project-203/code
```

## How to run the code
### Centralized setting
The example of running code in the centralized setting:
```
python3 -m main --dataset cifar10 --optimizer signmuon --data data --device cuda:1 --epochs 50 
```
#### Main Settings
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | *(Required)* | Dataset: `mnist`, `cifar10` |
| `--optimizer` | `signmuon` | Optimizer: `signmuon`, `muon`, `signsgd`, `sgd`, `adam` |
| `--data` | `./data` | Path to dataset directory |
| `--download` | `False` | Download dataset if missing |
| `--device` | `cpu` | Device for training (`cuda:0`, `cpu`, etc.) |
| `--seed` | `0` | Random seed for reproducibility |

#### Training Hyperparameters
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `128` | Batch size for data loader |
| `--lr` | `1e-3` | Learning rate for main optimizer parameters |
| `--lr-aux` | `1e-3` | Learning rate for auxiliary parameters |
| `--momentum` | `0.9` | momentum |
| `--nesterov` | `False` | Enable Nesterov momentum |

#### Muon/SignMuon Specific Parameters
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--ns-steps` | `5` | Number of Newton--Schulz iterations|
| `--lambda-mult` | `1.0` | Step size multiplier for Muon/SignMuon |

#### Other
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--weight-decay` | `5e-4` | Weight decay (Used for Adam/SGD; ignored by Muon/SignMuon internal logic if not specified) |
| `--run-name` | `""` | Folder name for saving results in `./saves/` (auto-generated if empty) |

### Federated setting
The example of running code in the federated setting:
```
python3 -m federated_main --model cnn2 --dataset cifar10 --algorithm signmuon --rounds 2000 --n_parties 10 --n_steps 3 --batch_size 64 --device cuda:3 --eval_freq 100
```
#### Main Settings
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | `cifar10` | Dataset: `mnist`, `cifar10` |
| `--download`| `False` |	Download dataset automatically if not found locally |
| `--model` | `cnn2` | Model architecture: `cnn2`, `resnet9` |
| `--algorithm` | `signmuon` | Optimizer: `signmuon`, `muon`, `signsgd`, `sgd`, `adam` |
| `--device` | `cpu` | Device for training (`cuda:0`, `cpu`, etc.) |
| `--seed` | `0` | Random seed for reproducibility |

#### Federated Learning Hyperparameters
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--rounds` | `10` | Number of communication rounds |
| `--n_parties` | `10` | Number of clients |
| `--n_steps` | `5` | Number of local optimizer steps per client per round |
| `--batch_size` | `32` | Batch size on each client |
| `--lr` | `1e-3` | Learning rate |
| `--momentum` | `0.9` | Momentum |

#### Muon/SignMuon Specific Parameters
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--ns_steps` | `5` | Number of Newton--Schulz iterations |
| `--lambda_mult` | `1.0` | Step size multiplier |
| `--no_norm_weight` | `False` | Disable weight normalization in Muon/SignMuon |

#### Data Partitioning
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--partition` | `homo` | Data distribution type: `homo` (IID), `noniid-labeldir` (Non-IID) |
| `--beta` | `0.5` | Dirichlet concentration parameter for Non-IID (smaller = more Non-IID) |

#### Other
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--eval_freq` | `1` | Model evaluation frequency on test set (every N rounds) |
| `--run_name` | `""` | Folder name for saving results (auto-generated if empty) |
| `--weight_decay` | `5e-4` | Weight decay (Adam only) |
| `--eps` | `1e-8` | Epsilon (Adam only) |

### Results and Graphs
The results of each run are saved to the `saves_federated/<run_name>/` folder (`saves/<run_name>/` for the centralized setting).

Inside the folder you will find:
*   `metrics.json`: History of metrics (accuracy, loss) over rounds.
*   `model_global.pt`: Global model weights after the final round.

To plot the results, use the Jupyter Notebook `federated_plot.ipynb` (or `plot.ipynb`).
