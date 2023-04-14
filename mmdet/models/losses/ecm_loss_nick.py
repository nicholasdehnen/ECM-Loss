import warnings

import torch
import torch.nn as nn
import numpy as np

from .accuracy import accuracy
from ..builder import LOSSES
from ...utils.logger import get_root_logger


@LOSSES.register_module()
class EffectiveClassMarginLossNick(nn.Module):

    def __init__(self,
                 stats_file = None,
                 loss_weight = 1.0,
                 **kwargs):
        """Effective class margin loss.
        Reimplemented by Nicholas Dehnen, according to the ECCV22 paper by Cho & Krähenbühl.

        Args:
            stats_file (file, path-like): File to load class distribution from. Defaults to None.
            reduction (str, optional): Loss reduction method. Defaults to 'mean'. Supports: 'mean', 'sum', 'none'
            loss_weight (float, optional): Optional loss weight. Defaults to 1.0.

        Raises:
            RuntimeError: Throws when class statistics aren't given. These are required for ECM-Loss calculation.
        """
        super(EffectiveClassMarginLossNick, self).__init__()
        self.logger = get_root_logger()
        self.rfuncs = {'mean': np.mean, 'sum': np.sum, 'none': lambda x: x}
        
        # load class distribution (required for margin calculation)
        if stats_file is None:
            error_message = "ECM-Loss needs class distribution to work!\n" + \
            "Please provide a text file containing the number of positive samples for each class.\n" + \
            "See class_stats_example.txt for an example."
            raise RuntimeError(error_message)
        else:
            self.class_stats = np.loadtxt(stats_file, comments=['#'], dtype='int')
            self.logger.info(f'ECM-Loss: Loaded distribution for {self.class_stats.shape[0]} classes.')
        
        # complementary arrays of positive and negative examples for each class
        n_pos = self.class_stats
        n_neg = np.sum(self.class_stats) - n_pos
        
        # upper bound for ranking error
        # pg. 7, theorem 2: negative-to-positive ratio
        a_c = n_neg / n_pos
        
        # pg. 8, theorem 2: linear approximation of bound
        self.m_c = a_c * np.log(1 + 1. / a_c) # simplification
        self.m_c = torch.cat((torch.tensor(self.m_c), torch.ones(1))) # add 1, nop for bg class (see forward fn)
        
        # pre-calculate margins and weights
        # pg. 9, eq. 8: calculation of tightest margins
        n_neg_r, n_pos_r = n_neg ** 1./4., n_pos ** 1./4.
        gamma_pos = n_neg_r / (n_pos_r + n_neg_r)
        gamma_neg = n_pos_r / (n_pos_r + n_neg_r)
        
        # pg. 10, eq. 11: calculate weights as inverse/reciprocal of margins
        self.weight_pos = torch.tensor(1. / gamma_pos)
        self.weight_neg = torch.tensor(1. / gamma_neg)
        
        # below is just taken from the default loss implementations in mmdet
        self.loss_weight = loss_weight
        self.custom_accuracy = True
    
    
    # pg. 10, eq. 11: surrogate scoring function (shifted sigmoid)
    def score(self, x):
        e_x = torch.exp(x)
        return (self.weight_pos * e_x) / (self.weight_pos * e_x + self.weight_neg * torch.exp(-x))
    

    def forward(self,
                cls_score,
                label,
                label_weights=None,
                **kwargs):
        """Forward function.
        """
        # warn on un-used inputs
        if label_weights != None:
            warnings.warn('label_weights was not None, but ECM loss does not implement support label weights!')
        
        # get device
        device = cls_score.device
        
        # ensure all tensors are on correct device (nop if already on device)
        self.weight_pos = self.weight_pos.to(device)
        self.weight_neg = self.weight_neg.to(device)
        self.m_c = self.m_c.to(device)
        
        # binary mask from target labels
        mask = torch.zeros_like(cls_score, device=device)
        mask[torch.arange(len(label)), label] = 1
        
        # pg. 10, eq. 10: surrogate effective class-margin loss
        
        # scoring broken down into score_a (scoring function from paper) and score_b (regular BCE for bg class)
        score_a = self.score(cls_score[:, :-1])
        score_b = torch.sigmoid(cls_score[:, -1])
       
        # concatenate scores back together
        score = torch.cat((score_a, score_b[:, None]), dim=1)
        
        # masked scores
        masked_pos_score = mask * torch.log(score)
        masked_neg_score = (1 - mask) * torch.log(1 - score)
        
        # calculate loss
        ecm_sum = torch.sum(self.m_c * (masked_pos_score + masked_neg_score))
        l_ecm = - ecm_sum / cls_score.shape[0]
        
        return self.loss_weight * l_ecm


    # taken from author's code, solely for metrics calculation, not paper related
    def get_accuracy(self, cls_score, labels):
        pos_inds = labels < self.num_classes
        acc = dict()
        obj_labels = (labels == self.num_classes).long() # 0 fg, 1 bg
        acc_objectness = accuracy(torch.cat([1 - cls_score[:, -1:], cls_score[:, -1:]], dim=1), obj_labels)
        acc_classes = accuracy(cls_score[:, :-1][pos_inds], labels[pos_inds])
        
        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes 
        return acc
