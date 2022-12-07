import torch
from torch import nn


class MatchingLoss(nn.Module):
    def __init__(self, beta: float = 10):
        """
            from this paper: https://arxiv.org/abs/1711.10485
            beta is gamma_3
        """
        super().__init__()
        self.beta = beta
        self.eps = 1e-9

    def forward(self, x, y, labels):
        mask = self.create_mask(labels)
        sim = self.cosine_sim(x, y)
        filled = mask * sim

        loss_1 = torch.diag(sim) / filled.sum(dim=1)
        loss_2 = torch.diag(sim) / filled.sum(dim=0)

        loss = loss_1.log().sum() + loss_2.log().sum()
        return -loss

    def create_mask(self, labels):
        mask = torch.ne(labels.view(1, -1), labels.view(-1, 1))
        mask = mask.fill_diagonal_(1)
        mask = mask.to(dtype=torch.float, device=labels.device)
        return mask

    def cosine_sim(self, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        norm_x = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        norm_y = torch.linalg.vector_norm(y, ord=2, dim=-1, keepdim=True)

        num = torch.bmm(x, y.transpose(1, 2))
        den = torch.bmm(norm_x, norm_y.transpose(1, 2))

        sim = self.beta * (num / den.clamp(min=1e-8))
        return sim.exp().squeeze()


class DistinctiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, cls_x, cls_y, labels):
        return self.crit(cls_x, labels) + self.crit(cls_y, labels)


class SENLoss(nn.Module):
    def __init__(self, beta: int = 10):
        super().__init__()
        self.matching_loss = MatchingLoss(beta)
        self.distinctive_loss = DistinctiveLoss()

    def forward(self, x, y, cls_x, cls_y, labels):
        match_loss = self.matching_loss(x, y, labels)
        dist_loss = self.distinctive_loss(cls_x, cls_y, labels)
        return match_loss, dist_loss, match_loss + dist_loss
