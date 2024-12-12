import os
import torch

class Ckpt():
    def __init__(self, net, tag="notag", base_path="checkpoints"):
        self.net = net
        self.path = os.path.join(base_path, str(tag))
        os.makedirs(self.path, exist_ok=True)


    def save(self, epoch):
        filename = os.path.join(self.path, "model_epoch%d.pth" % epoch)
        torch.save(self.net.state_dict(), filename)



