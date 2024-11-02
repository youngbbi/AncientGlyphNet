from torch import nn

from models.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, gamma=1.0, delta=1.0,  ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma  # 确保这里定义了gamma
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.character_bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)  # 使用BCEWithLogitsLoss
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]
        # 添加Character Map的损失计算
        character_maps = pred[:, 3, :, :]
        character_gt = batch['character_map']
        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        loss_character_maps = self.character_bce_loss(character_maps, character_gt)  # 计算字符损失
        metrics = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps, loss_character_maps=loss_character_maps)

        if pred.size()[1] > 3:  # 确保预测了binary map
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            # 总损失包含所有部分的损失和它们的权重
            loss_all = (self.alpha * loss_shrink_maps +
                        self.beta * loss_threshold_maps +
                        self.gamma * loss_binary_maps +
                        self.delta * loss_character_maps)
            metrics['loss'] = loss_all
        elif pred.size()[1] > 2:
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics
