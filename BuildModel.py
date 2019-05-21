from torch import nn
from torch import optim
from torchvision import models
from torch.optim import lr_scheduler
import math
import My_DetNet
def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr'] ]
    return lr
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
# 定义lr的变化规则
# schedules = [
# [[0, 0.01],[30, 0.0001],[40, 0.00001],[50, 0.00005]]
# ]
# 则 0~30 -> 0.01, 30~40 -> 0.0001, 40~50 -> 0.00001, 50~inf -> 0.00005
class ScheduleLR(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, schedules, last_epoch=-1):
        self.schedules = schedules
        # 添加上inf
        for g in range(len(self.schedules)):
            self.schedules[g].append([float("inf"), self.schedules[g][-1][1]])
        super(ScheduleLR, self).__init__(optimizer, last_epoch)
        if len(self.schedules) != len(optimizer.param_groups):
            raise ValueError("Expected {} schedules, but got {}".format(
                len(optimizer.param_groups), len(schedules)))

    def get_lr(self):
        lrs = []
        for (g, base_lr) in enumerate(self.base_lrs):
            schedule = self.schedules[g]
            for (i, [milestone, lr]) in enumerate(schedule):
                next_milestone = schedule[i+1][0]
                if milestone <= self.last_epoch < next_milestone:
                    lrs.append(lr)
                    break
        return lrs

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
# INPUT: ClassNumber
# OUTPUT: basic_model, optimizer, loss_fun
def build_model(ClassNum=10):
    basic_model = My_DetNet.VAE()
    loss_fun = My_DetNet.loss_func
    return basic_model, loss_fun

def build_optimizer(basic_model):
    optimizer = optim.SGD(basic_model.parameters(), lr=1e-8)
    a = basic_model.parameters()
    # optimizer = optim.SGD(basic_model.classifier.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = ScheduleLR(optimizer, schedules=[
        [[0, 1e-6], [40, 1e-6], [100, 1e-6]]
    ])

    return optimizer, scheduler