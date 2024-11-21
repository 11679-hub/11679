from .FCENet import FCENet
import torch
from torch import nn
from collections import OrderedDict

def get_model():
    model = FCENet()
    return model

def load_model(model):
    net = get_model()
    checkpoint = torch.load(model)
    net.cuda()
    try:
        net.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    return net


if __name__ == "__main__":
    from thop import profile
    from thop import clever_format

    model = get_model()
    # model = get_pretrain()
    input = torch.randn(1, 3, 128, 128)
    ir = torch.randn(1, 1, 128, 128)
    flops, params = profile(model, inputs=(input, ir))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)