import scipy.sparse as sp
import numpy as np
import json, os, pickle
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils import scipy_sparse_mat_to_torch_sparse_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _Dataset(object):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number

        Requirements:
            ids of students, questions, concepts all renumbered
        """
        self._raw_data = data
        self._concept_map = concept_map
        # reorganize datasets
        self._data = {}
        # example: {0:{20: 1}}
        for sid, qid, label in data:
            self._data.setdefault(int(sid), {})
            self._data[sid].setdefault(int(qid), {})
            self._data[sid][qid] = label

        self.n_students = num_students
        self.n_questions = num_questions
        self.n_concepts = num_concepts

    @property
    def num_students(self):
        return self.n_students

    @property
    def num_questions(self):
        return self.n_questions

    @property
    def num_concepts(self):
        return self.n_concepts

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data

    @property
    def concept_map(self):
        return self._concept_map

class JOBDataset(_Dataset, Dataset):
    def __init__(self, data, skill_map, num_geeks, num_jobs, num_skills):
        """
        Args:
            data: list, [(geek_id, job_id, score)]
            skill_map: dict, concept map {job_id: skill_id}
            num_geeks: int, total geek number
            num_jobs: int, total job number
            num_skills: int, total skill number

        Requirements:
            ids of geeks, jobs, skills all renumbered
        """
        super().__init__(data, skill_map, num_geeks, num_jobs, num_skills)


    def __getitem__(self, item):
        geek_id, job_id, score = self._raw_data[item]
        geek_id = int(geek_id)
        job_id = int(job_id)
        
        skills = np.array([0.5] * 3*self.n_concepts)
        
        for i in range(3):
            start_id = i * self.n_concepts
            end_id = min(start_id + self.n_concepts, 3*self.n_concepts)
            skills[start_id:end_id][[int(i) for i in self.concept_map[str(job_id)][i]]] = 1.
            # list_i = self.concept_map[str(job_id)][i]
            # skills[i][[int(i) for i in list_i]] = 1.
        
        return geek_id, job_id, score, skills

    def __len__(self):
        return len(self._raw_data)


class RecDataset(Dataset):
    def __init__(self, data, skill_map, num_geeks, num_jobs, num_skills):
        super(RecDataset, self).__init__()
        self.data = data
        self.concept_map = skill_map
        self.n_concepts = num_skills
        self._data = []
        for row in data:
            self._data.append([row[0], [row[1]]+eval(row[2])])

    def __getitem__(self, idx):
        geek_id, pos_neg_list = self._data[idx]
        geek_ids = [int(geek_id) for i in range(len(pos_neg_list))]
        skill_list = []
        for job_id in pos_neg_list:
            skills = np.array([0.5] * 3*self.n_concepts)
            for i in range(3):
                start_id = i * self.n_concepts
                end_id = min(start_id + self.n_concepts, 3*self.n_concepts)
                skills[start_id:end_id][[int(i) for i in self.concept_map[str(job_id)][i]]] = 1.
            
            skill_list.append(skills.tolist())

        return np.array(geek_ids), np.array(pos_neg_list), np.array(skill_list)

    def __len__(self):
        return len(self._data)


class MyDataset(object):

    def __init__(self, train_path, valid_path, test_path, train_rec_path, valid_rec_path, test_rec_path, job_skill_path, num_geeks, num_jobs, num_skills):
        """
        :param train_path: train data path
        :param valid_path: valid data path
        :param test_path: test data path
        :param train_rec_path: train rec data path
        :param valid_rec_path: valid rec data path
        :param test_rec_path: test rec data path
        :param job_skill_paht: job-skill dict path
        :param num_geeks: the num of geeks
        :param num_jobs: the num of jobs
        :param num_skills: the num of skills
        """
        # 1、read datas from csv files
        train_data = np.array(pd.read_csv(train_path))
        valid_data = np.array(pd.read_csv(valid_path))
        test_data = np.array(pd.read_csv(test_path))
        train_rec_data = np.array(pd.read_csv(train_rec_path))
        valid_rec_data = np.array(pd.read_csv(valid_rec_path))
        test_rec_data = np.array(pd.read_csv(test_rec_path))
        job_skill = json.load(open(job_skill_path, 'r'))

        # 2、construct the Dataset class
        self.train_data = JOBDataset(train_data, job_skill, num_geeks, num_jobs, num_skills)
        self.valid_data = JOBDataset(valid_data, job_skill, num_geeks, num_jobs, num_skills)
        self.test_data = JOBDataset(test_data, job_skill, num_geeks, num_jobs, num_skills)
        self.train_rec_data = RecDataset(train_rec_data, job_skill, num_geeks, num_jobs, num_skills)
        self.valid_rec_data = RecDataset(valid_rec_data, job_skill, num_geeks, num_jobs, num_skills)
        self.test_rec_data = RecDataset(test_rec_data, job_skill, num_geeks, num_jobs, num_skills)

    def get_dataloader(self, batch_size):
        """
        :param batch_size: batch size
        :return:
        """
        train_data_loader = DataLoader(self.train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        valid_data_loader = DataLoader(self.valid_data,
                                       batch_size=batch_size,
                                       shuffle=True)                            
        test_data_loader = DataLoader(self.test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
        train_rec_data_loader = DataLoader(self.train_rec_data,
                                      batch_size=batch_size,
                                      shuffle=True)
        valid_rec_data_loader = DataLoader(self.valid_rec_data,
                                      batch_size=batch_size,
                                      shuffle=True)                            
        test_rec_data_loader = DataLoader(self.test_rec_data,
                                      batch_size=batch_size,
                                      shuffle=True)

        return train_data_loader, train_rec_data_loader, valid_data_loader, valid_rec_data_loader,test_data_loader, test_rec_data_loader


def inter2coo_matrix(inter_matrix, geek_num, job_num):
    """
    :param inter_matrix:
    """
    pos_inter = inter_matrix[inter_matrix[:, 2] > 1]
    user_num, item_num = geek_num, job_num
    train = sp.coo_matrix((np.ones(pos_inter.shape[0]), (pos_inter[:, 0], pos_inter[:, 1])), shape=(user_num, item_num))
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
    
    train = train.tocoo()

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)
    # svd
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=5)
    u_mul_s = svd_u @ (torch.diag(s))  # [num_user, 5] * [5, 5]
    v_mul_s = svd_v @ (torch.diag(s))  # [num_item, 5] * [5, 5]


    pos_inter = inter_matrix[inter_matrix[:, 2] > 1]
    user_num, item_num = geek_num, job_num
    pos_inter[:, 1] = pos_inter[:, 1] + user_num
    pos_inter_converse = pos_inter.copy()
    pos_inter_converse[:, [0, 1]] = pos_inter_converse[:, [1, 0]]
    pos_inter = np.vstack((pos_inter, pos_inter_converse))
    
    train = sp.coo_matrix((np.ones(pos_inter.shape[0]), (pos_inter[:, 0], pos_inter[:, 1])), shape=(user_num+item_num, user_num+item_num))
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
    
    train = train.tocoo()

    ego_adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)

    return {"adj_norm":adj_norm, "ego_adj_norm":ego_adj_norm, "u_mul_s":u_mul_s, "v_mul_s":v_mul_s, "ut":svd_u.T, "vt":svd_v.T}


def dict2matrix(vec_dict):
    sorted_values = [value for _, value in sorted(vec_dict.items(), key=lambda x: int(x[0]))]
    matrix = torch.nn.Embedding.from_pretrained(torch.FloatTensor(sorted_values))
    return matrix


def dataset4train(dataset_name, data_config, batch_size):
    """
    :param dataset_name:
    :param data_config:
    :param batch_size:
    :return:
    """
    data_config = data_config[dataset_name]
    folds = data_config["folds"]  # k-fold

    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []
    train_rec_loader_list = []
    valid_rec_loader_list = []
    test_rec_loader_list = []
    geek_job_graph = []

    for i in range(folds):
        Data = MyDataset(
            os.path.join(data_config["dpath"], data_config[f"train_data"]),
            os.path.join(data_config["dpath"], data_config[f"valid_data"]),
            os.path.join(data_config["dpath"], data_config[f"test_data"]),
            os.path.join(data_config["dpath"], data_config[f"train_rec_data"]),
            os.path.join(data_config["dpath"], data_config[f"valid_rec_data"]),
            os.path.join(data_config["dpath"], data_config[f"test_rec_data"]),
            os.path.join(data_config["dpath"], data_config[f"job_skill_dict"]),
            data_config["num_geek"],
            data_config["num_job"],
            data_config["num_skill"]
        )

        train_loader, train_rec_loader, valid_loader, valid_rec_loader, test_loader, test_rec_loader = Data.get_dataloader(batch_size)
        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)
        test_loader_list.append(test_loader)
        train_rec_loader_list.append(train_rec_loader)
        valid_rec_loader_list.append(valid_rec_loader)
        test_rec_loader_list.append(test_rec_loader)

    geek_num = data_config["num_geek"]
    job_num = data_config["num_job"]
    train_data_csv = np.array(pd.read_csv(os.path.join(data_config["dpath"], data_config[f"train_data"])))
    geek_job_graph.append(list(train_data_csv[:, 0]))
    geek_job_graph.append(list(train_data_csv[:, 1]+geek_num))
    svd_dict = inter2coo_matrix(train_data_csv, geek_num, job_num)

    geek_vec_dict_path = os.path.join(data_config["dpath"], data_config[f"geek_vec_dict"])
    job_vec_dict_path = os.path.join(data_config["dpath"], data_config[f"job_vec_dict"])
    with open(geek_vec_dict_path, 'rb') as file:
        geek_vec_dict = pickle.load(file)
    with open(job_vec_dict_path, 'rb') as file:
        job_vec_dict = pickle.load(file)

    geek_vec_matrix = dict2matrix(geek_vec_dict)
    job_vec_matrix = dict2matrix(job_vec_dict)
    
    
    return train_loader_list, train_rec_loader_list, valid_loader_list, valid_rec_loader_list, test_loader_list, test_rec_loader_list, torch.LongTensor(geek_job_graph), svd_dict, geek_vec_matrix, job_vec_matrix