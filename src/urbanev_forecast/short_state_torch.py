"""Predictive relaxation and direct bounded MLPs; not identified physical rates."""
import math
import torch
from torch import nn


def relaxation_curve(p,mu,rho,steps):
    jj=torch.arange(1,steps+1,device=p.device,dtype=p.dtype)[None,:]
    retention=rho[:,None]**jj
    return retention*p[:,None]+(1-retention)*mu[:,None]


class StateNet(nn.Module):
    def __init__(self,input_dim,relaxation,seed=0):
        super().__init__();self.relaxation=relaxation
        self.first=nn.Linear(input_dim,32);self.second=nn.Linear(32,32);self.head=nn.Linear(32,2 if relaxation else 12)
        torch.manual_seed(seed)
        for layer in (self.first,self.second):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if relaxation:
            with torch.no_grad():self.head.bias[1]=math.log(9)
    def forward(self,x,p):
        hidden=torch.tanh(self.second(torch.tanh(self.first(x))));r=self.head(hidden)
        anchor=torch.logit(p.clamp(1e-4,1-1e-4))
        if self.relaxation:
            mu=torch.sigmoid(anchor+r[:,0]);rho=torch.sigmoid(r[:,1])
            return relaxation_curve(p,mu,rho,12)
        return torch.sigmoid(anchor[:,None]+r)
