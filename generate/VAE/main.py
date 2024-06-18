from dataset import dataloader
from loss import vae_loss
import torch
from torch import optim
from vae import VAE
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import argparse
import numpy as np
import os
import shutil

# 设置运行设备是GPU还是CPU
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
# 设置默认参数
parser = argparse.ArgumentParser(description="Variational Auto-Encoder MNIST Example")
parser.add_argument('--result_dir', type=str, default='./output', metavar='DIR', help='output directory')
parser.add_argument('--save_dir', type=str, default='./checkPoint', metavar='N', help='model saving directory')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of training times')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--test_every', type=int, default=10, metavar='N', help='test after every epochs')
parser.add_argument('--num_worker', type=int, default=0, metavar='N', help='the number of workers')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--z_dim', type=int, default=20, metavar='N', help='the dim of latent variable')
parser.add_argument('--input_dim', type=int, default=28*28, metavar='N', help='image size')
parser.add_argument('--input_channel', type=int, default=1, metavar='N', help='input channel')
args = parser.parse_args()
kwargs = {'num_workers':2, 'pin_memory':True} if cuda else {}

def save_checkpoint(state, is_best, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)

def test(model, optimizer, test_data, epoch, best_test_loss):
    test_avg_loss = 0
    with torch.no_grad():
        for test_batch_index, (test_x, _) in enumerate(test_data):
            test_x = test_x.to(device)
            # 前向传播
            test_x_hat, test_mu, test_log_var = model(test_x)

            test_loss, test_BCE, test_KLD = vae_loss(test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss
        
        test_avg_loss /= len(test_data.dataset)
        """测试隐变量"""
        z = torch.randn(args.batch_size, args.z_dim).to(device)
        # 重构隐变量
        random_res = model.decode(z).view(-1,1,28,28)
        save_image(random_res, './%s/random_sampled-%d.png' %(args.result_dir, epoch+1))
        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_dir)

    return best_test_loss


def main():
    minist_train, minist_test, classes = dataloader(args.batch_size, args.num_worker)
    print(len(minist_train),len(minist_test), classes)
    model = VAE(z_dim=args.z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_test_loss = np.finfo('f').max
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' %args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' %args.resume)
        else:
            print('=> no checkpoint founad at %s' %args.resume)
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    loss_epoch = []
    for epoch in range(start_epoch, args.epochs):
        loss_batch = []
        for batch_index, (x, _) in enumerate(minist_train):
            x = x.to(device)

            # 前向传播
            x_hat, mu, log_var = model(x)
            loss, BCE, KLD = vae_loss(x_hat, x, mu, log_var)
            loss_batch.append(loss.item())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每100个batch打印出statistics
            if(batch_index + 1) % 100 == 0:
                print('Epoch[{}/{}], Batch[{}/{}] : Total-loss={:.4f}, BCE-Loss = {:.4f}, KLD-loss={:.4f}'
                        .format(epoch + 1, args.epochs, batch_index + 1, len(minist_train.dataset)// args.batch_size,
                          loss.item() / args.batch_size, BCE.item() / args.batch_size ,
                           KLD.item() / args.batch_size ))

            if batch_index == 0:

                x_concat = torch.cat([x.view(-1, 1, 28, 28), x_hat.view(-1, 1, 28, 28)])
                save_image(x_concat, './%s/reconstructed-%d.png'%(args.result_dir, epoch + 1))

        loss_epoch.append(np.sum(loss_batch)/len(minist_train.dataset))

        # 测试模型
        if(epoch + 1) % args.test_every == 0:
            best_test_loss = test(model, optimizer, minist_test, epoch, best_test_loss)

    return loss_epoch


if __name__ == '__main__':
    loss_epoch = main()
    # 绘图
    plt.plot(loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()