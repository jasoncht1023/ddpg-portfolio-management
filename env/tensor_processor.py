import torch as T
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tucker

class TensorProcessor(nn.Module):
    def __init__(self, n_actions):
        super(TensorProcessor, self).__init__()
        self.tucker_dimension = [8, 6, 6, 6]
        self.n_actions = n_actions
        self.relu = nn.ReLU()
        self.conv3d = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 3, 1))
        tl.set_backend("pytorch")

    def forward(self, x):
        x = self.conv3d(x)
        x = self.relu(x)
        core, factors = tucker(x, rank=self.tucker_dimension)  # can be change
        core.requires_grad_(True)
        x = T.flatten(core)
        return x

