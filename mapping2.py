import torch

model_four_para_dict = {
    "DBCNN_LIVE": (57.6545, 23.6798),
    "DBCNN_CSIQ": (0.5046, 0.2212),
    "DBCNN_TID": (4.5363, 2.0833),
    "UNIQUE_LIVE": (-0.4212, 1.7329),
    "UNIQUE_CSIQ": (-0.7479, 1.3099),
    "UNIQUE_TID": (1.0418, 2.7986),
    "TReS_LIVE": (77.4899, 31.3389),
    "TReS_CSIQ": (0.6495, 0.2610),
    "TReS_TID": (6.1354, 3.2399),
    "LIQE_LIVE":(2.5015, 1.3928),
    "LIQE_CSIQ":(2.1409, 1.0923),
    "LIQE_TID":(3.3926, 2.9342),
                    }



A = 10.0
B = 0.0 
def logistic_mapping(x, key):
    C, S = model_four_para_dict[key]
    z = (x - C) / S
    return (A - B) / (1 + torch.exp(-z)) + B