import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from lib.models.layers.utils import soft_get


class Hierarchical_Task_Learning:
    def __init__(self, epoch0_loss, stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term: self.index2term.index(term) for term in self.index2term}  # term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses = []
        self.loss_graph = {'seg_loss': [],
                           'size2d_loss': [],
                           'offset2d_loss': [],
                           'offset3d_loss': ['size2d_loss', 'offset2d_loss'],
                           'size3d_loss': ['size2d_loss', 'offset2d_loss'],
                           'heading_loss': ['size2d_loss', 'offset2d_loss'],
                           'depth_loss': ['size2d_loss', 'size3d_loss', 'offset2d_loss']}

    def compute_weight(self, current_loss, epoch):
        T = 140
        # compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term]) == 0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device)
                # update losses list
        if len(self.past_losses) == self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2] - past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1 - (mean_diff / self.init_diff).relu().unsqueeze(0)

            time_value = min(((epoch - 5) / (T - 5)), 1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic]) != 0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]
                    loss_weights[current_topic] = time_value ** (1 - control_weight)
                    if loss_weights[current_topic] != loss_weights[current_topic]:
                        for pre_topic in self.loss_graph[current_topic]:
                            print('NAN===============', time_value, control_weight,
                                  c_weights[0][self.term2index[pre_topic]], pre_topic, self.term2index[pre_topic])
            # pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)

        return loss_weights

    def update_e0(self, eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class DIDLoss(nn.Module):
    def __init__(self, epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch

    def forward(self, preds, targets):

        if targets['mask_2d'].sum() == 0:
            bbox2d_loss = 0
            bbox3d_loss = 0
            self.stat['offset2d_loss'] = 0
            self.stat['size2d_loss'] = 0
            self.stat['depth_loss'] = 0
            self.stat['offset3d_loss'] = 0
            self.stat['size3d_loss'] = 0
            self.stat['heading_loss'] = 0
        else:
            bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
            bbox3d_loss = self.compute_bbox3d_loss(preds, targets)

        seg_loss = self.compute_segmentation_loss(preds, targets)

        mean_loss = seg_loss + bbox2d_loss + bbox3d_loss
        return float(mean_loss), self.stat

    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss

    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss

        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')

        loss = offset2d_loss + size2d_loss
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss

    def compute_bbox3d_loss(self, input, target, mask_type='mask_2d'):
        bs = target['depth'].shape[0]
        batch_idx = input['batch_idx'][input['train_tag']]
        num_objects = input['num_objects']
        if len(num_objects.shape) > 0:
            batch_idx[num_objects[0]:] += bs / 2
        roi_surface_mask_target = extract_target_from_tensor(target['roi_surface_masks'], target[mask_type])
        roi_surface_net_mask_target = extract_target_from_tensor(target['roi_surface_net_masks'], target[mask_type])
        roi_h2d = input['roi_h2d'][input['train_tag']]
        roi_h3d = input['roi_h3d'][input['train_tag']]
        att_depth = input['att_depth'][input['train_tag']]
        roi_h2d_uncer = input['roi_h2d_uncer'][input['train_tag']]
        roi_h3d_uncer = input['roi_h3d_uncer'][input['train_tag']]
        att_depth_uncer = input['att_depth_uncer'][input['train_tag']]

        surf_depth_net = input['surf_depth_net'][input['train_tag']]
        surf_depth_net_uncer = input['surf_depth_net_uncer'][input['train_tag']]

        ins_depth_net = input['ins_depth_net'][input['train_tag']]
        ins_depth_net_uncer = input['ins_depth_net_uncer'][input['train_tag']]
        grd_depth = input['grd_depth']
        grd_depth_offset = input['grd_depth_offset'][input['train_tag']]

        roi_h2d_target = extract_target_from_tensor(target['roi_h2ds'], target[mask_type])
        roi_h3d_target = extract_target_from_tensor(target['roi_h3ds'], target[mask_type])
        att_depth_target = extract_target_from_tensor(target['att_depths'], target[mask_type])
        ins_depth_net_target = extract_target_from_tensor(target['depth'], target[mask_type])
        surf_depth_target = extract_target_from_tensor(target['surf_depths'], target[mask_type])

        roi_cord_2d = extract_target_from_tensor(target['roi_cord_2ds'], target[mask_type])
        roi_cord_2d[..., 1] += roi_h2d
        grd_depth_points_pred = []
        for j in range(bs):
            roi_cord_2d_this_batch = roi_cord_2d[batch_idx == j].view(-1, 2)
            grd_depth_points_pred_this_batch, _ = soft_get(grd_depth[j], roi_cord_2d_this_batch, True, True)
            grd_depth_points_pred.append(grd_depth_points_pred_this_batch.view(-1, 2, 5, 5))
        grd_depth_points_pred = torch.cat(grd_depth_points_pred, 0)
        surf_depth = grd_depth_points_pred[:, 0, :, :] + (1. / (grd_depth_offset.sigmoid() + 1e-6) - 1.)
        surf_depth_uncer = input['surf_depth_uncer'][input['train_tag']]
        surf_depth_loss = laplacian_aleatoric_uncertainty_loss(surf_depth[roi_surface_mask_target],
                                                              surf_depth_target[roi_surface_mask_target],
                                                              surf_depth_uncer[roi_surface_mask_target])
        surf_depth_net_loss = laplacian_aleatoric_uncertainty_loss(surf_depth_net[roi_surface_net_mask_target],
                                                               surf_depth_target[roi_surface_net_mask_target],
                                                               surf_depth_net_uncer[roi_surface_net_mask_target])

        ins_depth_grd = surf_depth + att_depth
        ins_depth_grd_uncer = torch.logsumexp(torch.stack([surf_depth_uncer, att_depth_uncer], -1), -1)
        ins_depth_grd_loss = laplacian_aleatoric_uncertainty_loss(ins_depth_grd.view(-1, 5 * 5),
                                                                  ins_depth_net_target.repeat(1, 5 * 5),
                                                                  ins_depth_grd_uncer.view(-1, 5 * 5))

        grd_depth_target = extract_target_from_tensor(target['grd_depths'], target[mask_type])

        grd_cord_2d_target = extract_target_from_tensor(target['grd_cord_2ds'], target[mask_type])
        grd_mask_target = extract_target_from_tensor(target['grd_masks'], target[mask_type])
        grd_depth_points_pred = []
        grd_uncer_points_pred = []
        grd_depth_target_valid = []
        for i in range(bs):
            grd_depth_target_valid_this_batch = grd_depth_target[batch_idx == i][grd_mask_target[batch_idx == i]]

            grd_cord_2d_target_valid = grd_cord_2d_target[batch_idx == i][grd_mask_target[batch_idx == i]]
            grd_depth_points_pred_this_batch, grd_depth_points_valid_this_batch = soft_get(grd_depth[i], grd_cord_2d_target_valid)
            grd_depth_points_pred.append(grd_depth_points_pred_this_batch[0])
            grd_uncer_points_pred.append(grd_depth_points_pred_this_batch[1])
            grd_depth_target_valid_this_batch = grd_depth_target_valid_this_batch[grd_depth_points_valid_this_batch]
            grd_depth_target_valid.append(grd_depth_target_valid_this_batch)

        grd_depth_points_pred = torch.cat(grd_depth_points_pred, 0)
        grd_uncer_points_pred = torch.cat(grd_uncer_points_pred, 0)
        grd_depth_target_valid = torch.cat(grd_depth_target_valid, 0)
        grd_depth_loss = laplacian_aleatoric_uncertainty_loss(grd_depth_points_pred,
                                                              grd_depth_target_valid,
                                                              grd_uncer_points_pred)


        roi_h2d_loss = F.l1_loss(roi_h2d, roi_h2d_target, reduction='mean') * 2 / 3 \
                       + laplacian_aleatoric_uncertainty_loss(roi_h2d[roi_surface_mask_target],
                                                              roi_h2d_target[roi_surface_mask_target],
                                                              roi_h2d_uncer[roi_surface_mask_target]) / 3
        roi_h3d_loss = F.l1_loss(roi_h3d, roi_h3d_target, reduction='mean') * 2 / 3 \
                       + laplacian_aleatoric_uncertainty_loss(roi_h3d[roi_surface_mask_target],
                                                              roi_h3d_target[roi_surface_mask_target],
                                                              roi_h3d_uncer[roi_surface_mask_target]) / 3
        att_depth_loss = laplacian_aleatoric_uncertainty_loss(att_depth[roi_surface_net_mask_target],
                                                              att_depth_target[roi_surface_net_mask_target],
                                                              att_depth_uncer[roi_surface_net_mask_target])
        # 2->1
        ins_depth_net_loss = laplacian_aleatoric_uncertainty_loss(ins_depth_net.view(-1, 5 * 5),
                                                                  ins_depth_net_target.repeat(1, 5 * 5),
                                                                  ins_depth_net_uncer.view(-1, 5 * 5))

        depth_loss = (grd_depth_loss + att_depth_loss + surf_depth_loss + surf_depth_net_loss) + (ins_depth_grd_loss + ins_depth_net_loss)

        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]
        offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')

        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']]
        size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
        size3d_loss = F.l1_loss(size3d_input, size3d_target, reduction='mean')
        size3d_loss = size3d_loss + 0.1 * roi_h2d_loss + roi_h3d_loss

        # compute heading loss
        heading_loss = compute_heading_loss(input['heading'][input['train_tag']],
                                            target[mask_type],  ## NOTE
                                            target['heading_bin'],
                                            target['heading_res'])

        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss

        if depth_loss != depth_loss:
            print('badNAN----------------depth_loss', depth_loss)
            print(roi_h2d_loss, roi_h3d_loss, grd_depth_loss, att_depth_loss, roi_surface_mask_target.sum())
        if offset3d_loss != offset3d_loss:
            print('badNAN----------------offset3d_loss', offset3d_loss)
        if size3d_loss != size3d_loss:
            print('badNAN----------------size3d_loss', size3d_loss)
        if heading_loss != heading_loss:
            print('badNAN----------------heading_loss', heading_loss)

        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss

        return loss


### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]


# compute heading loss two stage style

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)  # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')

    return cls_loss + reg_loss


'''    

def compute_heading_loss(input, ind, mask, target_cls, target_reg):
    """
    Args:
        input: features, shaped in B * C * H * W
        ind: positions in feature maps, shaped in B * 50
        mask: tags for valid samples, shaped in B * 50
        target_cls: cls anns, shaped in B * 50 * 1
        target_reg: reg anns, shaped in B * 50 * 1
    Returns:
    """
    input = _transpose_and_gather_feat(input, ind)   # B * C * H * W ---> B * K * C
    input = input.view(-1, 24)  # B * K * C  ---> (B*K) * C
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    input_cls, target_cls = input_cls[mask], target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    input_reg, target_reg = input_reg[mask], target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss
'''

if __name__ == '__main__':
    input_cls = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))
