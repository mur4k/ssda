from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# based on:
# https://kornia.readthedocs.io/en/latest/_modules/kornia/utils/one_hot.html

def one_hot(labels: torch.Tensor,
            num_classes: int,
            ignore_index: int = -1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    mask = labels.eq(ignore_index)
    labels_with_ignored_idx = torch.masked_fill(labels, mask, num_classes)
    shape = labels_with_ignored_idx.shape
    one_hot = torch.zeros(shape[0], num_classes+1, *shape[1:],
                          device=device, dtype=dtype)
    one_hot.scatter_(1, labels_with_ignored_idx.unsqueeze(1), 1.0)
    one_hot  = one_hot[:, :-1]
    return one_hot


# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'none', ignore_index: int = -1) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.ignore_index: int = ignore_index
        self.eps: float = 1e-8

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(\
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(labels=target, ignore_index=self.ignore_index, 
                                num_classes=input.shape[1], device=input.device, 
                                dtype=input.dtype)

        # compute the actual focal loss
        weight = torch.pow(1. - input_soft + self.eps, self.gamma)
        focal = - self.alpha * weight * torch.log(input_soft + self.eps)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                .format(self.reduction))
        return loss
    

class NT_Xent(nn.Module):

    def __init__(self, temperature, reduction='mean'):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_real_features: torch.Tensor, 
                image_perturbed_features: torch.Tensor, memory_bank_features: torch.Tensor):
        pos_similarities = F.cosine_similarity(image_real_features, 
                                image_perturbed_features, 1) / self.temperature
        neg_similarities = F.cosine_similarity(image_real_features.unsqueeze(1), 
                                memory_bank_features.unsqueeze(0), 2) / self.temperature
        neg_similarities = torch.sum(neg_similarities, dim=1)
        logits = torch.cat([pos_similarities.unsqueeze(-1), neg_similarities.unsqueeze(-1)], dim=(-1))
        labels = torch.zeros_like(pos_similarities, dtype=(torch.int64))
        loss_tmp = self.criterion(logits, labels)
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                .format(self.reduction))
        return loss
            
class NT_Xent2(nn.Module):

    def __init__(self, temperature, batch_size, reduction='mean'):
        super(NT_Xent2, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.batch_size = batch_size
        self.mask_positive = torch.diag_embed(torch.ones(batch_size, dtype=bool), offset=-batch_size) | \
            torch.eye(2*batch_size, dtype=bool) | \
            torch.diag_embed(torch.ones(batch_size, dtype=bool), offset=batch_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features: torch.Tensor, memory_bank_features: torch.Tensor):
        image_similarities = F.cosine_similarity(image_features.unsqueeze(1), 
                                image_features.unsqueeze(0), 2) / self.temperature
        memory_bank_similarities = F.cosine_similarity(image_features.unsqueeze(1), 
                                memory_bank_features.unsqueeze(0), 2) / self.temperature
        pos_similarities = torch.cat([image_similarities.diagonal(self.batch_size), 
                                image_similarities.diagonal(-self.batch_size)])
        neg_similarities = F.cosine_similarity(image_features.unsqueeze(1), 
                                memory_bank_features.unsqueeze(0), 2) / self.temperature
        neg_similarities = torch.sum(neg_similarities, dim=1)
        logits = torch.cat([pos_similarities.unsqueeze(-1), neg_similarities.unsqueeze(-1)], dim=(-1))
        labels = torch.zeros_like(pos_similarities, dtype=(torch.int64))
        loss_tmp = self.criterion(logits, labels)
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                .format(self.reduction))
        return loss


######################
# functional interface
######################


def focal_loss(input, target, alpha, gamma=2.0, reduction='none'):
    """Function that computes Focal loss.
    """
    return FocalLoss(alpha, gamma, reduction)(input, target)


def nt_xent(image_real_features, image_perturbed_features, 
            memory_bank_features, temperature=1.0, reduction='none'):
    """Function that computes Focal loss.
    """
    return NT_Xent(temperature, reduction)(image_real_features, 
                                           image_perturbed_features, memory_bank_features)


def nt_xent2(image_features, memory_bank_features, batch_size,
             temperature=1.0, reduction='none'):
    """Function that computes Focal loss.
    """
    return NT_Xent2(temperature, batch_size, reduction)(image_features, memory_bank_features)
