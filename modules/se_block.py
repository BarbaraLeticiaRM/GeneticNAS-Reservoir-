import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, n_channels, rate):
        super(SEBlock, self).__init__()
        self.se_block = nn.Sequential(nn.Linear(n_channels, int(n_channels / rate)),
                                      nn.ReLU(),
                                      nn.Linear(int(n_channels / rate), n_channels),
                                      nn.Sigmoid())

    # def forward(self, x):
    #     x_gp = torch.mean(torch.mean(x, dim=-1), dim=-1)
    #     att = self.se_block(x_gp).unsqueeze(dim=-1).unsqueeze(dim=-1)
    #     return x * att

    def forward(self, x):
        # Calcular a média global apenas na dimensão do comprimento
        x_gp = torch.mean(x, dim=-1)
        
        # Passar pela sequência SE
        att = self.se_block(x_gp)
        
        # Ajustar a atenção para a multiplicação com x
        att = att.unsqueeze(dim=-1)

        return x * att