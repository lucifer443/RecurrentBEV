import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss

from bev.registry import MODELS


@MODELS.register_module()
class BCELoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        """Binary cross entropy loss
    
        Args:
            reduction (stc, optional): . Defaults to 'mean'.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(BCELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self,
                input,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            input (torch.Tensor): The output of the network.
            target (torch.Tensor): The ground truth.
            weight (torch.Tensor, optional): Element-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_gt (int | None): The value of ground truth to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        assert target.dim() == input.dim()
        if weight is None:
            weight = torch.ones_like(target, dtype=torch.float)
        else:
            assert target.dim() == weight.dim()
        
        # weighted element-wise losses
        loss = self.loss_weight * F.binary_cross_entropy(
            input, target.float(), reduction='none')
        # do the reduction for the weighted loss
        loss = weight_reduce_loss(
            loss, weight, reduction=reduction, avg_factor=avg_factor)
        
        return loss
