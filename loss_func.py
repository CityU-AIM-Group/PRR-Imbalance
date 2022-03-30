import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class CELossWithLogits(nn.Module):
    """
    CE loss baseline
    """
    def __init__(self, class_counts: Union[list, np.array]):
        super(CELossWithLogits, self).__init__()
        class_counts = torch.FloatTensor(class_counts)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits, targets):
        targets = F.one_hot(targets, self.num_labels)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = torch.exp(logits)[:, None, :].sum(axis=-1) 

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()

class CPA_Loss_init(nn.Module):
    """
    Args:
    class_counts: The list of the number of samples for each class. 
    beta: Scale parameter to adjust the strength.
    """

    def __init__(self, class_counts: Union[list, np.array], beta: float = 0.8):
        super(CPA_Loss_init, self).__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** beta
        # print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits, targets, **kwargs):
        targets = F.one_hot(targets, self.num_labels)
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


class CPA_Loss(nn.Module):
    """
    Args:
    class_counts: The list of the number of samples for each class. 
    beta: Scale parameter to adjust the strength.
    """
    def __init__(
        self, 
        class_counts: Union[list, np.array], 
        beta: float = 0.8,
        clamp_thres: float = 0,
        tau: float = 3.0
        ):
        super(CPA_Loss, self).__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** beta
        # print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.global_factor = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6
        self.clamp_thres = clamp_thres
        self.tau = tau
      
    def proto_factor_cosine(self, source_proto, target_proto):
        """
        [C, D]: D is 64 or 4
        """
        # factor = 1
        norm_source = torch.norm(source_proto, dim=-1, keepdim=False)
        norm_target = torch.norm(target_proto.detach(), dim=-1, keepdim=False) # [C]
        factor_refined = torch.sum(source_proto*target_proto.detach(), dim=-1, keepdim=False)/(norm_source*norm_target+self.eps)
        return factor_refined # [C]
    
    def forward(self, logits, targets, local_proto, global_proto):
        targets = F.one_hot(targets, self.num_labels) # [N, C]
        self.global_factor = self.global_factor.to(targets.device) # [C, C]
        max_element, _ = logits.max(axis=-1)
        # [N, C]
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits) # [N, C]
        denominator = (
            (1 - targets)[:, None, :]
            * self.global_factor[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits) # [N, C]

        sigma = numerator / (denominator + self.eps) # [N, C]
        # proto factor
        cosine_score = self.proto_factor_cosine(source_proto=local_proto, target_proto=global_proto)
        proto_factor = (1+self.tau)/(cosine_score+self.tau) #
        # print(proto_factor)
        # sum in categories
        loss = (- proto_factor.view(1, -1) * targets * torch.log(sigma + self.eps)).sum(-1) # [N]
        return loss.mean() # scalar

#########################



def global_avg_proto(local_protos):
    # local_protos: client_num*C*D
    return torch.mean(local_protos, dim=0, keepdim=False) # C*D

def global_gaussian_proto(local_protos):
    # local_protos: client_num*C*D
    mean = torch.mean(local_protos, dim=0, keepdim=False)
    std = torch.clamp(
        torch.std(local_protos, dim=0, keepdim=False),
        min=1
        )
    sample = torch.randn(mean.shape).to(mean.device)
    return sample * std + mean # C*D

###########################
def proto_factor_cosine(local_proto, global_proto):
    """
    [C, D]: D is 64 or 4
    """
    # factor = 1
    norm_local = torch.norm(local_proto, dim=-1, keepdim=False)
    norm_global = torch.norm(global_proto, dim=-1, keepdim=False) # [C]
    factor_refined = torch.sum(local_proto*global_proto, dim=-1, keepdim=False)/(norm_local*norm_global+1e-6)
    return factor_refined # [C]

def tau_func(cosine_score, tau):
    proto_factor = (1+tau)/(cosine_score+tau) #
    return proto_factor