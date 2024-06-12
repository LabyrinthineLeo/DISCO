import os, sys
import torch
from torch.nn.functional import cross_entropy
import numpy as np
from sklearn.metrics import roc_auc_score
from .utils import getHR, getNDCG
import logging
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_forward(model, data, mode="point_wise"):
    """
    :param model: model
    :param data: data
    :param mode: default point_wise
    """

    if mode == "point_wise":
        geek_inputs, job_inputs, labels, skills = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device) 
    elif mode == "pair_wise":
        geek_inputs, pos_neg_inputs, skills = data[0].to(device), data[1].to(device), data[2].to(device)

    output, model_loss = model(geek_inputs, job_inputs, skills)  # [bsz, n_classes]
    loss = cross_entropy(output, labels) + model.beta * model_loss
    
    return loss


def evaluate(model, test_data, test_rec=None):
    """
    :param model: model
    :param test_data:
    :param test_rec:
    """
    model_name = model.model_name
    n_classes = model.n_classes
    class_list = [i for i in range(n_classes)]
    real = np.array([])
    pred = np.array([]).reshape(0, n_classes)
    hr5_list, ndcg5_list, hr10_list, ndcg10_list = [], [], [], []


    with torch.no_grad():
        # ====================for auc====================
        model.eval()
        for geek_inputs, job_inputs, labels, skills in test_data:
            geek_inputs, job_inputs, labels, skills = geek_inputs.to(device), job_inputs.to(device), labels.to(
                device), skills.to(device)
            labels = labels.float()
            skills = skills.float()

            output, model_loss = model(geek_inputs, job_inputs, skills, test=True)  # [bsz, n_classes]


            pred = np.vstack((pred, output.detach().cpu().numpy()))  # [bsz, n_classes]

            real = np.concatenate((real, labels.detach().cpu().numpy()))  # [bsz, ]

        model.train()

        if n_classes <= 2:
            pred_pos = pred[:, 1]
            auc = roc_auc_score(real, pred_pos)
        else:
            real_bin = label_binarize(real, classes=class_list)  # [N, n_classes]
            auc_values = []
            for i in range(n_classes):
                auc = roc_auc_score(real_bin[:, i], pred[:, i])
                auc_values.append(auc)
            auc = np.mean(auc_values)

        # ====================for hr&ndcg====================
        if test_rec:
            model.eval()
            for geek_inputs, pos_neg_inputs, skills in test_rec:
                # [bsz, n], [bsz, n], [bsz, n, skill_len]
                geek_inputs, pos_neg_inputs, skills = geek_inputs.to(device), pos_neg_inputs.to(device), skills.to(device)
                bsz = geek_inputs.shape[0]
                skills = skills.float()

                output, model_loss = model(geek_inputs, pos_neg_inputs, skills, test=True) # [bsz, n, n_classes]

                if n_classes == 4:
                    pred = np.sum(output.detach().cpu().numpy()[:,:,-2:], axis=-1)  # positive pred [bsz, n]
                else:
                    pred = output.detach().cpu().numpy()[:,:,1]
            # ================HR@k================
            hr5 = getHR(pred, 5)  # top-5
            hr10 = getHR(pred, 10)  # top-10
            hr5_list.append(hr5)
            hr10_list.append(hr10)
            # ================NDCG@k================
            ndcg5_list.extend(getNDCG(pred, 5))  # top-5
            ndcg10_list.extend(getNDCG(pred, 10))  # top-10

            model.train()

            hr5_final = np.mean(hr5_list)
            hr10_final = np.mean(hr10_list)
            ndcg5_final = np.mean(ndcg5_list)
            ndcg10_final = np.mean(ndcg10_list)
        
        if test_rec:
            return auc, hr5_final, ndcg5_final, hr10_final, ndcg10_final
        else:
            return auc


def train_model(model, train_loader, train_rec, valid_loader, valid_rec, test_loader, test_rec, num_epochs, opt, result_path, break_epoch=5):
    """
    :param model: 
    :param train_loader: 
    :param test_loader: 
    :param num_epochs: 
    :param opt: 
    :param result_path: 
    :param gj_graph: geek_job_graph
    :return:
    """
    min_rmse, max_auc, best_epoch, best_step = np.inf, 0, 0, 0
    max_hr5, max_ndcg5 = 0, 0
    max_hr10, max_ndcg10 = 0, 0
    train_step = 0
    save_model = True
    best_emb = None


    for i in range(1, num_epochs + 1):
        loss_mean = []
        for j, data in enumerate(train_loader):
            train_step += 1
            model.train()

            loss = model_forward(model, data)

            opt.zero_grad()
            loss.backward()  # compute gradients
            opt.step()  # update model’s parameters

            loss_mean.append(loss.detach().cpu().numpy())

        loss_mean = np.mean(loss_mean)
        
        trainauc = evaluate(model, train_loader)
     
        auc_epoch, hr5_epoch, ndcg5_epoch, hr10_epoch, ndcg10_epoch = evaluate(model, valid_loader, valid_rec)
        
        if auc_epoch >= max_auc:
            if save_model:
                torch.save(model.state_dict(), os.path.join(result_path, model.model_name + "_best.ckpt"))
                best_emb = model.E_ego
            max_auc = auc_epoch
            max_hr5 = hr5_epoch
            max_hr10 = hr10_epoch
            max_ndcg5 = ndcg5_epoch
            max_ndcg10 = ndcg10_epoch
            best_epoch = i
        
        torch.cuda.empty_cache()
        
        valid_auc, valid_hr5, valid_ndcg5, valid_hr10, valid_ndcg10 = auc_epoch, hr5_epoch, ndcg5_epoch, hr10_epoch, ndcg10_epoch

        print("="*20+f"Epoch {i}"+"="*20)
        logging.info("="*20+f"Epoch {i}"+"="*20)
        print(f"Epoch: {i}, train_auc: {trainauc:.4}, valid_auc: {valid_auc:.4}, valid_hr5: {valid_hr5:.5}, "
                f"valid_ndcg5: {valid_ndcg5:.5}, valid_hr10: {valid_hr10:.5}, valid_ndcg10: {valid_ndcg10:.5}, "
                f"best epoch: {best_epoch}, best auc: {max_auc:.4}, "
                f"best hr5: {max_hr5:.5}, best ndcg5: {max_ndcg5:.5}, best hr10: {max_hr10:.5}, best ndcg10: {max_ndcg10:.5}, "
                f"train loss: {loss_mean:.5}, model: {model.model_name}, save_dir: {result_path.split('_')[-1]}")
        logging.info(f"Epoch: {i}, train_auc: {trainauc:.4}, valid_auc: {valid_auc:.4}, valid_hr5: {valid_hr5:.5}, "
                f"valid_ndcg5: {valid_ndcg5:.5}, valid_hr10: {valid_hr10:.5}, valid_ndcg10: {valid_ndcg10:.5}, "
                f"best epoch: {best_epoch}, best auc: {max_auc:.4}, "
                f"best hr5: {max_hr5:.5}, best ndcg5: {max_ndcg5:.5}, best hr10: {max_hr10:.5}, best ndcg10: {max_ndcg10:.5}, "
                f"train loss: {loss_mean:.5}, model: {model.model_name}, save_dir: {result_path.split('_')[-1]}")

        if i - best_epoch >= break_epoch:
            break

    # Final Testing
    cpt_path = os.path.join(result_path, model.model_name + "_best.ckpt")
    cpt = torch.load(cpt_path)
    model.load_state_dict(cpt)
    model.E_ego = best_emb
    test_auc, test_hr5, test_ndcg5, test_hr10, test_ndcg10 = evaluate(model, test_loader, test_rec)
    print(f"Final Testing, test_auc: {test_auc:.4}, test_hr5: {test_hr5:.5}, "
            f"test_ndcg5: {test_ndcg5:.5}, test_hr10: {test_hr10:.5}, test_ndcg10: {test_ndcg10:.5}, ")
    logging.info(f"Final Testing, test_auc: {test_auc:.4}, test_hr5: {test_hr5:.5}, "
            f"test_ndcg5: {test_ndcg5:.5}, test_hr10: {test_hr10:.5}, test_ndcg10: {test_ndcg10:.5}, ")

    return test_auc, test_hr5, test_hr10, test_ndcg5, test_ndcg10, best_epoch