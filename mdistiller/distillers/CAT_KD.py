import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
    
class CAT_KD(Distiller):

    def __init__(self, student, teacher, cfg):
        super(CAT_KD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CAT_KD.LOSS.CE_WEIGHT
        self.CAT_loss_weight = cfg.CAT_KD.LOSS.CAT_loss_weight
        self.onlyCAT = cfg.CAT_KD.onlyCAT
        self.CAM_RESOLUTION = cfg.CAT_KD.LOSS.CAM_RESOLUTION
        self.relu = nn.ReLU()
        
        self.IF_NORMALIZE = cfg.CAT_KD.IF_NORMALIZE
        self.IF_BINARIZE = cfg.CAT_KD.IF_BINARIZE
        
        self.IF_OnlyTransferPartialCAMs = cfg.CAT_KD.IF_OnlyTransferPartialCAMs
        self.CAMs_Nums = cfg.CAT_KD.CAMs_Nums
        # 0: select CAMs with top x predicted classes
        # 1: select CAMs with the lowest x predicted classes
        self.Strategy = cfg.CAT_KD.Strategy
        
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)       
        tea = feature_teacher["feats"][-1]
        stu = feature_student["feats"][-1]
                        
        # perform binarization
        if self.IF_BINARIZE:
            n,c,h,w = tea.shape
            threshold = torch.norm(tea, dim=(2,3), keepdim=True, p=1)/(h*w)
            tea =tea - threshold
            tea = self.relu(tea).bool() * torch.ones_like(tea)
        
        
        # only transfer CAMs of certain classes
        if self.IF_OnlyTransferPartialCAMs:
            n,c,w,h = tea.shape
            with torch.no_grad():
                if self.Strategy==0:
                    l = torch.sort(logits_teacher, descending=True)[0][:, self.CAMs_Nums-1].view(n,1)
                    mask = self.relu(logits_teacher-l).bool()
                    mask = mask.unsqueeze(-1).reshape(n,c,1,1)
                elif self.Strategy==1:
                    l = torch.sort(logits_teacher, descending=True)[0][:, 99-self.CAMs_Nums].view(n,1)
                    mask = self.relu(logits_teacher-l).bool()
                    mask = ~mask.unsqueeze(-1).reshape(n,c,1,1)
            tea,stu = _mask(tea,stu,mask)

        loss_feat = self.CAT_loss_weight * CAT_loss(
            stu, tea, self.CAM_RESOLUTION, self.IF_NORMALIZE
        )
         
        if self.onlyCAT is False:
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            losses_dict = {
                "loss_CE": loss_ce,
                "loss_CAT": loss_feat,
            }
        else:
            losses_dict = {
                "loss_CAT": loss_feat,
            }

        return logits_student, losses_dict


def _Normalize(feat,IF_NORMALIZE):
    if IF_NORMALIZE:
        feat = F.normalize(feat,dim=(2,3))
    return feat

def CAT_loss(CAM_Student, CAM_Teacher, CAM_RESOLUTION, IF_NORMALIZE):   
    CAM_Student = F.adaptive_avg_pool2d(CAM_Student, (CAM_RESOLUTION, CAM_RESOLUTION))
    CAM_Teacher = F.adaptive_avg_pool2d(CAM_Teacher, (CAM_RESOLUTION, CAM_RESOLUTION))
    loss = F.mse_loss(_Normalize(CAM_Student, IF_NORMALIZE), _Normalize(CAM_Teacher, IF_NORMALIZE))
    return loss
    

def _mask(tea,stu,mask):
    n,c,w,h = tea.shape
    mid = torch.ones(n,c,w,h).cuda()
    mask_temp = mask.view(n,c,1,1)*mid.bool()
    t=torch.masked_select(tea, mask_temp)
    
    if (len(t))%(n*w*h)!=0:
        return tea, stu

    n,c,w_stu,h_stu = stu.shape
    mid = torch.ones(n,c,w_stu,h_stu).cuda()
    mask = mask.view(n,c,1,1)*mid.bool()
    stu=torch.masked_select(stu, mask)
    
    return t.view(n,-1,w,h), stu.view(n,-1,w_stu,h_stu)