import torch.nn as nn
import torch


class DiceLoss(nn.Module):

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        self.epsilon = 1e-12
        

        # predict = predict.detach()
        # predict.requires_grad=True


        pre = predict.flatten()
        tar = target.flatten()

        
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score
