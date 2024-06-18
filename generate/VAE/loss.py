import torch.nn.functional as F
import torch

def vae_loss(x_hat, x, mu, log_var):
    """计算vae的损失函数(交叉熵+相对熵(KL散度))

    Args:
        x_hat (_type_): predict x
        x (_type_): real x
        mu (_type_): 均值
        log_var (_type_): 方差的对数
    return: 
        loss:损失
        BCE:交叉熵
        KL:相对熵
    """

    # 交叉熵损失
    BCE = F.binary_cross_entropy(x_hat,x,reduction='sum')

    # KL散度
    KL = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

    # 损失
    loss = BCE + KL
    return loss, BCE, KL