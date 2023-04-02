import torch
def load_param(model, weight_path):
        
    print("Resume from checkpoint {}".format(weight_path))
    param_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
    if 'state_dict' in param_dict.keys():
        param_dict = param_dict['state_dict']
        
        
    print('ignore_param:')
    print([k for k, v in param_dict.items() if k not in model.state_dict() or
                    model.state_dict()[k].size() != v.size()])
    print('unload_param:')
    print([k for k, v in model.state_dict().items() if k not in param_dict.keys() or
                    param_dict[k].size() != v.size()] )

    param_dict = {k: v for k, v in param_dict.items() if k in model.state_dict() and
                        model.state_dict()[k].size() == v.size()}
    for i in param_dict:
        model.state_dict()[i].copy_(param_dict[i])
from model import build_model
from config import cfg
cfg.merge_from_file("configs/mfa.yml")
a = build_model(cfg)
load_param(a,'C:\\Users\\PC\\Downloads\\weights\\deeplabv2_Mean_B_step6.pth')