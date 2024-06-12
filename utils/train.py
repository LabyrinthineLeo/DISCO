import os, sys
sys.path.append("..")

import argparse, logging
import json

import torch
from torch.optim import SGD, Adam
from .utils import set_seed, debug_print, save_config
from .dataset import dataset4train
from .init_model import init_model
from .train_model import train_model
import datetime
import numpy as np

device = "cpu" if not torch.cuda.is_available() else "cuda"


def main(params):
    # ====================1.====================

    if params['seed_true']:
        set_seed(params["seed"])
    model_name, dataset_name, fold, save_dir = params["model_name"], \
                                               params["dataset_name"], \
                                               params["folds"], \
                                               params["save_dir"]

    debug_print(text="load config files.", fuc_name="main")


    # ====================2.====================
    train_config = {
        "batch_size": params["batch_size"],
        "num_epochs": params["epoch"],
        "optimizer": "adam"
    }

    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open("../configs/data_info.json") as fin:
        data_config = json.load(fin)


    # ====================3.====================
    print("=" * 20 + "start init data" + "=" * 20)
    print(dataset_name, model_name)
    for i, j in data_config.items():
        print(f"{i}: {j}")
    print(fold, batch_size)


    # ====================4.====================
    debug_print(text="init_dataset", fuc_name="main")
    train_loader_list, train_rec_loader_list, valid_loader_list, valid_rec_loader_list, test_loader_list, test_rec_loader_list, geek_job_graph, svd_dict, geek_vec_matrix, job_vec_matrix = dataset4train(dataset_name, data_config, batch_size)
    geek_job_graph = geek_job_graph.to(device)
    geek_vec_matrix = geek_vec_matrix.to(device)
    job_vec_matrix = job_vec_matrix.to(device)


    # ====================5.====================
    params_str = "-".join([str(v) for k, v in params.items() if not k in ['other_config']])

    time_suffix = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    result_path = os.path.join(save_dir, params_str+f"-{time_suffix}")
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    # logging
    start_time = datetime.datetime.now().timestamp()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        filename=os.path.join(result_path, 'log.txt'),
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='[%(asctime)s %(levelname)s] %(message)s',
    )

    print("=" * 20 + "print params" + "=" * 20)
    logging.info("=" * 20 + "print params" + "=" * 20)
    # print(f"params: {params}")
    for i, j in params.items():
        print(f"{i}: {j}")
        logging.info(f"{i}: {j}")

    print("=" * 20 + "training model" + "=" * 20)
    logging.info("=" * 20 + "training model" + "=" * 20)
    print(f"Start training model: {model_name}, \nsave_dir: {result_path}, \ndataset_name: {dataset_name}")
    logging.info(f"Start training model: {model_name}, \nsave_dir: {result_path}, \ndataset_name: {dataset_name}")
    print("=" * 20 + "model config" + "=" * 20)
    logging.info("=" * 20 + "model config" + "=" * 20)
    print("=" * 20 + "train config" + "=" * 20)
    logging.info("=" * 20 + "train config" + "=" * 20)
    print(f"train_config: {train_config}")
    logging.info(f"train_config: {train_config}")


    save_config(train_config, data_config[dataset_name], params, result_path)
    learning_rate = params["lr"]

    debug_print(text="init_model", fuc_name="main")


    # ====================6.====================

    print("=" * 20 + "init model" + "=" * 20)
    logging.info("=" * 20 + "init model" + "=" * 20)
    print(f"model_name:{model_name}")
    logging.info(f"model_name:{model_name}")


    model = init_model(model_name, data_config[dataset_name], params, svd_dict=svd_dict)
        
    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)


    # ====================7.====================
    validauc, validacc = -1, -1
    testauc, testacc = -1, -1
    best_epoch = -1
    save_model = False

    debug_print(text="train model", fuc_name="main")

    
    test_auc_list = []
    test_hr5_list, test_hr10_list = [], []
    test_ndcg5_list, test_ndcg10_list = [], []
    test_loader, test_rec = test_loader_list[0], test_rec_loader_list[0]
    for i, (train_loader, train_rec, valid_loader, valid_rec) in enumerate(zip(train_loader_list, train_rec_loader_list, valid_loader_list, valid_rec_loader_list)):
        print("=" * 20 + "training time:{}".format(i) + "=" * 20)
        test_auc, test_hr5, test_hr10, test_ndcg5, test_ndcg10, best_epoch = train_model(model, train_loader, train_rec, valid_loader, valid_rec, test_loader, test_rec, num_epochs, opt, result_path, 5)

        print("=" * 20 + "New Fold" + "=" * 20)
        logging.info("=" * 20 + "New Fold" + "=" * 20)

        model = init_model(model_name, data_config[dataset_name], params, svd_dict=svd_dict)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
  
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)


        test_auc_list.append(test_auc)
        test_hr5_list.append(test_hr5)
        test_hr10_list.append(test_hr10)
        test_ndcg5_list.append(test_ndcg5)
        test_ndcg10_list.append(test_ndcg10)


    print("=" * 20 + "auc list" + "=" * 20)
    print(["{:.5}".format(i) for i in test_auc_list])
    logging.info(["{:.5}".format(i) for i in test_auc_list])

    print("=" * 20 + "hr5 list" + "=" * 20)
    print(["{:.5}".format(i) for i in test_hr5_list])
    logging.info(["{:.5}".format(i) for i in test_hr5_list])

    print("=" * 20 + "ndcg5 list" + "=" * 20)
    print(["{:.5}".format(i) for i in test_ndcg5_list])
    logging.info(["{:.5}".format(i) for i in test_ndcg5_list])

    print("=" * 20 + "hr10 list" + "=" * 20)
    print(["{:.5}".format(i) for i in test_hr10_list])
    logging.info(["{:.5}".format(i) for i in test_hr10_list])

    print("=" * 20 + "ndcg10 list" + "=" * 20)
    print(["{:.5}".format(i) for i in test_ndcg10_list])
    logging.info(["{:.5}".format(i) for i in test_ndcg10_list])


    print("=" * 20 + "the mean and std of auc/hr5/ndcg5/hr10/ndcg10" + "=" * 20)
    print("auc mean:{:.5}, auc std:{:.5}".format(np.mean(test_auc_list), np.std(test_auc_list)))
    logging.info("auc mean:{:.5}, auc std:{:.5}".format(np.mean(test_auc_list), np.std(test_auc_list)))

    print("hr5 mean:{:.5}, hr5 std:{:.5}".format(np.mean(test_hr5_list), np.std(test_hr5_list)))
    logging.info("hr5 mean:{:.5}, hr5 std:{:.5}".format(np.mean(test_hr5_list), np.std(test_hr5_list)))

    print("ndcg5 mean:{:.5}, ndcg5 std:{:.5}".format(np.mean(test_ndcg5_list), np.std(test_ndcg5_list)))
    logging.info("ndcg5 mean:{:.5}, ndcg5 std:{:.5}".format(np.mean(test_ndcg5_list), np.std(test_ndcg5_list)))

    print("hr10 mean:{:.5}, hr10 std:{:.5}".format(np.mean(test_hr10_list), np.std(test_hr10_list)))
    logging.info("hr10 mean:{:.5}, hr10 std:{:.5}".format(np.mean(test_hr10_list), np.std(test_hr10_list)))

    print("ndcg10 mean:{:.5}, ndcg10 std:{:.5}".format(np.mean(test_ndcg10_list), np.std(test_ndcg10_list)))
    logging.info("ndcg10 mean:{:.5}, ndcg10 std:{:.5}".format(np.mean(test_ndcg10_list), np.std(test_ndcg10_list)))


    print(f"start:{now}")
    print(f"end:{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    end_time = datetime.datetime.now().timestamp()
    print(f"cost time:{(end_time-start_time)//60} min")