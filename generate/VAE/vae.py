from torch import nn
import torch
import torch.nn.functional as F

class VAE(nn.Module):

    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # encoder      [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim,z_dim) # mu (均值)
        self.fc3 = nn.Linear(h_dim,z_dim) # log_var （协方差）

        # decoder      [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        """前向传播

        Args:
            x (_type_): 训练时的模型的输入[b, batch_size, 1, 28, 28]
        returns: 模型训练后的输出、mu和log_var
        """
        batch_size = x.shape[0] # 每一批含有的样本个数
        x = x.view(batch_size, self.input_dim)  # 将[b, batch_size, 1, 28, 28] => [b, batch_size, 784]

        # encoder
        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)
        x_hat = x_hat.view(batch_size, 1, 28, 28)

        return x_hat, mu, log_var

    def encode(self, x):
        """将输入的图像编码

        Args:
            x (_type_): 输入的图像

        Returns:
            _type_: mu和log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var
    
    def reparameterization(self, mu, log_var):
        """让采样出的均值和协方差符合高斯分布

        Args:
            mu (_type_): _description_
            log_var (_type_): _description_

        Returns:
            _type_: _description_
        """
        sigma = torch.exp(log_var*0.5)
        eps = torch.randn_like(sigma)

        return mu + sigma * eps
    
    def decode(self, z):
        h = F.relu(self.fc4(z))
        x_hat = self.fc5(h)

        return x_hat