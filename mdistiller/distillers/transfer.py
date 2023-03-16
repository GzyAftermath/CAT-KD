import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
    
class transfer(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
    src code: https://github.com/szagoruyko/attention-transfer
    """

    def __init__(self, student, cfg):
        super(transfer, self).__init__()
        self.student = student
        l = len([x for x in student.parameters()])
        for i, param in enumerate(student.parameters()):
            if i < l-2:
                param.requires_grad = False

        
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        loss_ce = F.cross_entropy(logits_student, target)

        losses_dict = {
            "loss_ce": loss_ce,
        }

        return logits_student, losses_dict

    
    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters() if v.requires_grad]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0


    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])