import math
from typing import Tuple
import torch
import torch.nn.functional as F


def mixup_criterion(pred, y_emo, y_neu, lam) -> torch.Tensor:
    w_emo = math.sqrt(1)
    w_neu = math.sqrt(1)
    l_emo = lam * F.cross_entropy(pred, y_emo)
    l_neu = (1 - lam) * F.cross_entropy(pred, y_neu)
    loss = (w_emo * l_emo + w_neu * l_neu) / (w_emo + w_neu)
    return torch.mean(loss)


def rank_loss(ri, rj, lam_diff) -> torch.Tensor:
    p_hat_ij = F.sigmoid(ri - rj)
    rank_loss = torch.mean(- lam_diff @ torch.log(p_hat_ij) - (1 - lam_diff) @ torch.log(1 - p_hat_ij)) / lam_diff.size(0)
    return rank_loss


class EmotionIntensityLoss:
    alpha = 0.1
    beta = 1.0
    enable_mixup_loss = True
    enable_rank_loss = True
    
    def __call__(
        self, 
        predi, predj, y_emo, y_neu, lam_i, lam_j
    ) -> Tuple[torch.Tensor, dict]:
        ii, hi, ri = predi
        ij, hj, rj = predj
        
        lam_diff = (lam_i - lam_j) / 2 + 0.5
        
        losses = {}
        if self.enable_mixup_loss:
            mixup_loss_i = mixup_criterion(hi, y_emo, y_neu, lam_i)
            mixup_loss_j = mixup_criterion(hj, y_emo, y_neu, lam_j)
            mixup_loss = mixup_loss_i + mixup_loss_j
            losses.update({
                "mi": mixup_loss_i.item(), 
                "mj": mixup_loss_j.item(),
            })
        else:
            mixup_loss = torch.tensor(0.)
        
        if self.enable_rank_loss:
            ranking_loss = rank_loss(ri, rj, lam_diff)
            losses.update({
                "rank": ranking_loss.item(),
            })
        else:
            ranking_loss = torch.tensor(0.)
        
        total_loss = self.alpha * mixup_loss + self.beta * ranking_loss
        losses.update({
            "total": total_loss.item(),
        })
        
        return total_loss, losses


if __name__ == "__main__":
    criterion = EmotionIntensityLoss()
    loss, losses = criterion(
        (torch.randn(10, 10), torch.randn(10, 10), torch.randn(10, 10)),
        (torch.randn(10, 10), torch.randn(10, 10), torch.randn(10, 10)),
        torch.randint(0, 10, (10,)),
        torch.randint(0, 10, (10,)),
        torch.rand(10),
        torch.rand(10),
    )
    print(loss)
    print(losses)