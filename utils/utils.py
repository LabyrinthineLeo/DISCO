import os, datetime, json
import random as python_random
from torch import nn
import copy, math
import pandas as pd
import numpy as np
import torch

device = "cpu" if not torch.cuda.is_available() else "cuda"

def set_seed(seed):
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e_s:
        print("Set seed failed, details are ", e_s)
        pass

    np.random.seed(seed)
    python_random.seed(seed)

    # cuda env
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def get_now_time():
    """
    Return the time string, the format is %Y-%m-%d %H:%M:%S
    :return:
    """
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string


def debug_print(text, fuc_name=""):
    """
    Printing text with function name.
    :param text:
    :param fuc_name:
    :return:
    """
    print("="*20+"print info"+"="*20)
    print(f"{get_now_time()}_{fuc_name}, said: {text}")


def save_config(train_config, data_config, params, save_dir):
    # d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    d = {"train_config": train_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout, ensure_ascii=False, indent=2)


def bpr_loss(pos_pred, neg_pred):
    """
    pos_pred: [bsz, n]
    neg_pred: [bsz, n]
    """
    loss = torch.mean((pos_pred - neg_pred -1) ** 2)
    return loss


def getHR(batch_pred, k):
    """
    :param batch_pred: pred scores, [bsz, n], n is the len of candidate list
    :param k: top k
    """
    bsz = batch_pred.shape[0]
    rank_index = np.argsort(batch_pred)[:,::-1][:, :k]  # top-k
    pos_index = np.zeros((bsz, 1))
    # ================HR@k================
    is_hit = np.any(pos_index==rank_index, axis=1)  # [bsz, ]
    return np.mean(is_hit)


def getNDCG(batch_pred, k):
    """
    ref: https://github.com/LianHaiMiao/Attentive-Group-Recommendation
    :param batch_pred: pred scores, [bsz, n], n is the len of candidate list
    :param k: top k
    :return: the ndcg value list
    """
    bsz, len_cand = batch_pred.shape
    rank_index = np.argsort(batch_pred)[:,::-1][:, :k]
    ndcg_list = []
    for each_list in rank_index:
        flag = 0
        for i in range(k):
            item = each_list[i]
            if item == 0:  # positive index
                flag = 1
                value = math.log(2) / math.log(i+2)
                break
        if not flag:
            value = 0
        ndcg_list.append(value)
    return ndcg_list


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    :param sparse_mx: [(user, item, score), (), ...]
    :return:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # [2, n_inter]
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # [n_inter, ]
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def rmse_score(a, b):
    return np.sqrt(np.mean((a-b)**2))


def mae_score(a,b):
    return np.mean(np.abs(a-b))


def cal_meanandstd(x):
    mean = np.mean(x)
    std = np.std(x)
    print(mean, std)


def cosin_similarity(x, y):
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))


def generate_qmatrix(data_config, gamma=0.0):
    df_train = pd.read_csv(os.path.join(data_config["dpath"], "train_test_seqs_grp.csv"))[["questions", "concepts"]]
    df_test = pd.read_csv(os.path.join(data_config["dpath"], "train_test_seqs_stu.csv"))[["questions", "concepts"]]
    df = pd.concat([df_train, df_test])

    problem2skill = dict()
    for i, row in df.iterrows():
        qids, cids = [], []
        qids_ori = [[int(i) for i in sublist.split(",")] for sublist in row['questions'].split("|")]
        cids_ori = [[int(i) for i in sublist.split(",")] for sublist in row['concepts'].split("|")]
        for sub_q, sub_c in zip(qids_ori, cids_ori):
            for i, q in enumerate(sub_q):
                if q != -1:
                    qids.append(q)
                    cids.append(sub_c[i])

        assert len(qids) == len(cids), "the qid size must match the cid size!"

        for q, c in zip(qids, cids):
            if q in problem2skill:
                problem2skill[q].append(c)
            else:
                problem2skill[q] = [c]
    print("num ques:{}, num cpts:{}".format(len(set(qids)), len(set(cids))))
    n_problem, n_skill = data_config["num_q"], data_config["num_c"]
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        for c in problem2skill[p]:
            q_matrix[p][c] = 1
    np.savez(os.path.join(data_config["dpath"], "qmatrix.npz"), matrix=q_matrix)
    return q_matrix


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.2):
        """
        ref: https://github.com/LianHaiMiao/Attentive-Group-Recommendation/blob/master/model/agree.py
        :param embedding_dim: dimension of embedding
        :param drop_ratio: drop_ratio
        """
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight


class AttentionModule(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.2):
        """
        ref: https://github.com/LianHaiMiao/Attentive-Group-Recommendation/blob/master/model/agree.py
        :param embedding_dim: dimension of embedding
        :param drop_ratio: drop_ratio
        """
        super(AttentionModule, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        :param x: [N, n_span, n_dim]
        :return:
        """
        x_avg = torch.mean(x, dim=1)
        out = self.linear(x_avg)
        weight = torch.softmax(out.unsqueeze(-1), dim=0)
        output = torch.sum(weight * x, dim=0, keepdim=True)
        a = torch.mean(x, dim=0, keepdim=True)
        return output


class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=-1, keepdim=True)
        return x


if __name__ == '__main__':
    pass
