import os, sys
sys.path.append("..")

import argparse
from utils.train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="edu_rec")
    parser.add_argument("--model_name", type=str, default="ngcf_disco")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--seed_true", type=int, default=0)

    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--num_dim", type=int, default=256)

    args = parser.parse_args()

    params = vars(args)
    main(params)
