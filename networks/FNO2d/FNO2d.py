import torch.nn as nn
import torch.nn.functional as F
from neuralacoustics.utils import openConfig

from neuralop.layers.embeddings import GridEmbedding2D
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.padding import DomainPadding
from neuralop.layers.channel_mlp import ChannelMLP

#VIC this is the content of: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d_time.py
# i made a small modification to the original code, highlighted by the following comment: #VIC-mod

################################################################
# fourier layer
################################################################

class Permute(nn.Module):
    """
    simple module to perform permutations inside sequential networks
    """
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)

def fnoToSeq(n_layers, fno):
    """
    simple function to transform from FNOBlocks to nn.Sequential
    """
    return nn.Sequential(*[fno[i] for i in range(n_layers)])

class FNO2d(nn.Module):
    # WYNN-mod: Add stacks_num input argument
    def __init__(self, config_path, t_in):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.lift.
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.project.
        
        input: the solution of the previous t_in timesteps + 2 locations (u(t-t_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=t_in+2)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        # Parse config file
        network_config = openConfig(config_path, __file__)
        self.modes1 = network_config['network_parameters'].getint('network_modes')
        self.modes2 = network_config['network_parameters'].getint('network_modes')
        self.width = network_config['network_parameters'].getint('network_width')
        self.stacks_num = network_config['network_parameters'].getint('stacks_num')
        self.pad_on = network_config['network_parameters'].getboolean('pad_on')
        self.padding = network_config['network_parameters'].getfloat('padding')
        self.mlp_on = network_config['network_parameters'].getboolean('mlp_on')
        self.normalization_on = network_config['network_parameters'].getboolean('normalization_on')
        self.mlp_first = network_config['network_parameters'].getboolean('mlp_first')
        self.mlp_last = network_config['network_parameters'].getboolean('mlp_last')
        self.fact_on = network_config['network_parameters'].getboolean('fact_on')
        self.fact = network_config['network_parameters'].get('fact')

        self.grid_embed = nn.Sequential(
            Permute(0, 3, 1, 2),
            GridEmbedding2D(t_in+2),
        )
        
        #VIC-mod t_in is passed as parameter now, so that we can decide the number of input time steps
        if self.mlp_first:
            self.lift = ChannelMLP(
                in_channels=t_in+2,
                out_channels=self.width,
                hidden_channels=2*self.width,
                n_layers=2,
                n_dim=2,
                non_linearity=F.gelu,
            )
        else:
            self.lift = nn.Sequential(
                Permute(0, 2, 3, 1),
                nn.Linear(t_in+2, self.width),
                Permute(0, 3, 1, 2),
            )

        # WYNN-mod: A module list for stacking layers
        self.conv_list = fnoToSeq(self.stacks_num,
            FNOBlocks(
                in_channels=self.width,
                out_channels=self.width,
                n_modes=(self.modes1, self.modes2),
                n_layers=self.stacks_num,
                use_channel_mlp=self.mlp_on,
                channel_mlp_skip="soft-gating",
                norm="instance_norm" if self.normalization_on else None,
                factorization=self.fact if self.fact_on else None,
                non_linearity=F.gelu,
        ))

        if self.mlp_last:
            self.project = nn.Sequential(
                ChannelMLP(self.width, 1, self.width * 2),
                Permute(0, 2, 3, 1),
            )
        else:
            self.project = nn.Sequential(
                Permute(0, 2, 3, 1),
                nn.Linear(self.width, 128),
                nn.Linear(128, 1),
            )
            
        if self.pad_on:
            self.padder = DomainPadding(self.padding)

    def forward(self, x):
        x = self.grid_embed(x)

        x = self.lift(x)
        
        if self.pad_on:
            x = self.padder.pad(x) # pad the domain if input is non-periodic

        x = self.conv_list(x)
        
        if self.pad_on:
            x = self.padder.unpad(x) # pad the domain if input is non-periodic

        x = self.project(x)

        return x
