# -*- coding: utf-8 -*-  
'''
@author: zhjp   2022/5/28 下午3:35
@file: code_bk.py.py
'''

# 计算原型正交损失 & 原型样本损失
sempdelta = semproto[label_idx] - self.gauss_eyes[label_idx]
sempsample = semproto[label_idx] - cls_feat[label_idx]

orthT = self.loss_semorth.T
sampT = self.loss_semproto.T
semorth = F.kl_div(
    (semproto[label_idx] / orthT).log(),
    (self.gauss_eyes[label_idx] / orthT).detach(),
    reduction='none').mean(-1) * (orthT * orthT)
