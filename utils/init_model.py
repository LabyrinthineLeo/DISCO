import torch, os
from models.NGCF_DISCO import NGCF_DISCO


device = "cpu" if not torch.cuda.is_available() else "cuda"


def init_model(model_name, data_config, params, graph=None, svd_dict=None):
    model = NGCF_DISCO(data_config["num_geek"], data_config["num_job"], params["num_dim"], svd_dict["ego_adj_norm"], data_config["num_level"], data_config["leaf_dim"], params['beta'], n_classes=params['n_classes']).to(device)
    return model