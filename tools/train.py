from xtuner.tools.train import main as train
try:
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass
if __name__ == '__main__':
    train()
