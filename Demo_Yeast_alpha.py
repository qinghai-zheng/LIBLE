import torch.optim as optim
from tools.model import LabelEnhanceNet, GapEstimationNet, LIB_Encoder, LIB_Decoder
from tools.datasets import GetDataset
from sklearn.preprocessing import scale
import scipy.io as sio
from tools.measures import *
import torch.nn.functional as F


def train():
    data_f_d = sio.loadmat('./datasets/Yeast_alpha.mat')
    data_l = sio.loadmat('./datasets/Yeast_alpha_binary.mat')

    num_sample = len(data_f_d['features'].T[0])
    features = data_f_d['features']
    f_dim = len(data_f_d['features'][0])
    dis_label_gt = data_f_d['labels']
    d_dim = len(data_f_d['labels'][0])
    log_label = data_l['logicalLabel']

    features_tmp = scale(features)
    x_data = torch.from_numpy(features_tmp).float().to(device)
    log_label = torch.from_numpy(log_label).float().to(device)

    dvib_enc = LIB_Encoder(f_dim, args.h_dim)
    dvib_dec = LIB_Decoder(args.h_dim, d_dim)
    LE_Net = LabelEnhanceNet(args.h_dim, d_dim).to(device)
    LE_Net = torch.nn.DataParallel(LE_Net)
    Gap_Net = GapEstimationNet(args.h_dim).to(device)
    Gap_Net = torch.nn.DataParallel(Gap_Net)
    dataset = GetDataset(x_data, log_label)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    op_pre = optim.Adam(list(dvib_enc.parameters())+list(dvib_dec.parameters()), lr=args.lr[0])
    op = optim.Adam(list(LE_Net.parameters())+list(Gap_Net.parameters())+list(dvib_enc.parameters())+list(dvib_dec.parameters()), lr=args.lr[1])
    lr_s = torch.optim.lr_scheduler.StepLR(op, step_size=20, gamma=0.9)

    # pretraining
    for epoch_pre in range(args.epochs[0]):
        for batch_idx, (batch, log_l, idx) in enumerate(train_loader):
            pre_loss_all = 0
            mu_d, std_d = dvib_enc(batch)
            z = mu_d + mu_d * (torch.randn(mu_d.size()).to(device))
            batch_hat = dvib_dec(z)
            pre_loss_rec = F.cross_entropy(batch_hat, log_l).div(math.log(2))
            pre_info_loss = -0.5 * (1 + 2 * torch.log(std_d) - torch.pow(mu_d, 2) - torch.pow(std_d, 2)).sum(1).mean().div(
                math.log(2))
            pre_loss_dvib = pre_loss_rec + args.para_hyper[0] * pre_info_loss
            pre_loss_all = pre_loss_all + pre_loss_dvib

            op_pre.zero_grad()
            pre_loss_all.backward()
            op_pre.step()

            if batch_idx % args.log_interval == 0:
                print('Pretrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_pre, batch_idx * len(log_l.T[0]), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), pre_loss_all))

    # training for the whole framework
    for epoch in range(args.epochs[1]):
        for batch_idx, (batch, log_l, idx) in enumerate(train_loader):
            loss_all = 0
            mu_d, std_d = dvib_enc(batch)
            z = mu_d + mu_d * (torch.randn(mu_d.size()).to(device))
            batch_hat = dvib_dec(z)
            loss_rec = F.cross_entropy(batch_hat,log_l).div(math.log(2))
            info_loss = -0.5 * (1 + 2 * torch.log(std_d) - torch.pow(mu_d,2) - torch.pow(std_d,2)).sum(1).mean().div(math.log(2))
            loss_dvib = loss_rec + args.para_hyper[0]*info_loss

            d_pre = LE_Net(z)
            gap = Gap_Net(z)
            lost_obj = (log_l - d_pre)**2
            lost_obj = torch.mul(lost_obj, (0.5*torch.pow(gap, -2))) + torch.log(torch.abs(gap))
            lost_obj = lost_obj.mean(1, keepdim=True)
            loss_all = loss_all + loss_dvib + args.para_hyper[1]*lost_obj.mean()

            op.zero_grad()
            loss_all.backward()
            op.step()
            lr_s.step()

            if batch_idx % args.log_interval == 0:
                print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(log_l.T[0]), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss_all))

    mu_d, std_d = dvib_enc(x_data)
    z = mu_d + mu_d * (torch.randn(mu_d.size()).to(device))
    distri_pre = LE_Net(z)
    distri_pre_tmp = []
    distri_pre_tmp.extend(distri_pre.data.cpu().numpy())
    preds = softmax(distri_pre_tmp)

    dists = []
    dist1 = chebyshev(dis_label_gt, preds)
    dist2 = clark(dis_label_gt, preds)
    dist3 = canberra(dis_label_gt, preds)
    dist4 = kl_dist(dis_label_gt, preds)
    dist5 = cosine(dis_label_gt, preds)
    dist6 = intersection(dis_label_gt, preds)

    dists.append(dist1)
    dists.append(dist2)
    dists.append(dist3)
    dists.append(dist4)
    dists.append(dist5)
    dists.append(dist6)

    return distri_pre, dists


def softmax(d, t=1):
    for i in range(len(d)):
        d[i] = d[i]*t
        d[i] = np.exp(d[i])/sum(np.exp(d[i]))
    return d


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--h_dim', type=int, default=256, metavar='N',
                        help='input batch size for training [default: 2000]')
    parser.add_argument('--para_hyper', type=int, default=[10, 1], metavar='N',
                        help='input batch size for training [default: 2000]')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training [default: 2000]')
    parser.add_argument('--epochs', type=int, default=[200, 100], metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lr', type=float, default=[5e-3, 1e-3], metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='choose CUDA device [default: cuda:1]')
    parser.add_argument('--seed', '-seed', type=int, default=0,
                        help='random seed (default: 0)')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    device = torch.device(args.device if args.cuda else 'cpu')

    setup_seed(args.seed)

    distri_pre, dists = train()

    print(np.round(dists, 4))