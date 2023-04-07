

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


class CALCULATELoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_data", torch.zeros(1, 1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output, teacher_output, s_data, t_data, epoch):

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(12)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        loss_cls = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss_cls += loss.mean()
                n_loss_terms += 1
        loss_cls /= n_loss_terms
        self.update_center(teacher_output)


        s_d = s_data / self.student_temp

        # teacher centering and sharpening
        t_d = F.softmax((t_data - self.center_data) / temp, dim=-1)
        t_d = t_d.detach()  
        
        loss_data = torch.sum(-t_d * F.log_softmax(s_d, dim=-1), dim=-1)
        
        self.update_center_data(t_data)
        
        return loss_cls, loss_data.mean()

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


    @torch.no_grad()
    def update_center_data(self, teacher_output):
        batch_center = torch.sum( torch.mean(teacher_output, dim=1, keepdim=True), dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center_data = self.center_data * self.center_momentum + batch_center * (1 - self.center_momentum)

