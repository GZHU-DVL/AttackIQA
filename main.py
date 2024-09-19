import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torchvision.transforms as transforms
import scipy.stats
import numpy as np
from tqdm import tqdm
from TReS import TReS
from BaseCNN import BaseCNN
from DBCNN import DBCNN
import pandas as pd

import argparse
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from datetime import datetime
from mapping2 import logistic_mapping
import random
from LIQE import LIQE

dataset_path_dict = {
    "LIVE": path to LIVE dataset,
    "CSIQ": path to CSIQ dataset,
    "TID": path to TID2013 dataset
}
linear_rescale_dict = {
    "LIVE": (0.0, 114.4147),
    "CSIQ": (0.0, 1.0),
    "TID": (0.0, 9.0),
}


def linear_rescale(mos, args):
    min_mos, max_mos = linear_rescale_dict[args.dataset]
    rescaled_mos = min_mos + 10 * (mos - min_mos) / (max_mos - min_mos)
    return rescaled_mos


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        # print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()


def predict_quality_score(x, args):
    if args.model == 'DBCNN':
        s = model(x).squeeze(1)
    elif args.model == 'UNIQUE':
        s, _ = model(x)
    elif args.model == 'TReS':
        s, _ = model(x)
        s = s.squeeze()
    elif args.model == 'LIQE':
        s, _, _ = model(x)


    s = logistic_mapping(s, args.model + '_' + args.dataset)

    return s


def obj_function(s, mos):
    obj = torch.zeros_like(s)
    for i in range(s.shape[0]):
        if mos[i] > 5:
            obj[i] = s[i]
        else:
            obj[i] = -s[i]
    return obj


def random_sign(size):
    return torch.sign(-1 + 2 * torch.rand(size=size))


def obj_function(s, mos):
    obj = torch.zeros_like(s)
    for i in range(s.shape[0]):
        if mos[i] > 5:
            obj[i] = s[i]
        else:
            obj[i] = -s[i]
    return obj


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p



def attackIQA(dataset_loader, args):
    eps = args.eps
    ####################
    # 0x01: initialize #
    ####################
    logger.log('start init......')
    gt = []
    loss_mins = []
    x_pert_bests = []
    x_oris = []
    pred_s_purt_cur_bests = []
    for i, (image, mos) in enumerate(dataset_loader):
        x = image.to(device)
        x_oris.append(x)
        gt.append(mos.cpu().numpy())
        f_x = fx[i]

        c, h, w = x.shape[1:]

        n_features = c * h * w
        init_delta = (eps) * random_sign(size=[x.shape[0], c, 1, w]).to(device)

        x_pert = torch.clamp(x + init_delta, 0, 1)

        pred_s = predict_quality_score(x_pert, args)

        loss_min = obj_function(pred_s, f_x)
        loss_mins.append(loss_min)
        x_pert_bests.append(x_pert)
        pred_s_purt_cur_bests.append(pred_s.cpu().numpy())
    logger.log('loss_mins = {}'.format(loss_mins))
    ####################
    # 0x01: optimize   #
    ####################

    logger.log('start optimize......')

    for i_iter in tqdm(range(args.n_queries - 2)):

        for n in range(args.n_sample // args.bs):
            # a batch of images: x_oris[n]
            x_ori, x_pert_curr, f_x = x_oris[n].clone(), x_pert_bests[n].clone(), fx[n]
            loss_min_curr = loss_mins[n]
            deltas_cur = x_pert_curr - x_ori

            p = p_selection(p_init=args.p_init, it=i_iter, n_iters=args.n_queries)
            # images in one batch
            # sample new delta for each img at random position
            for i_img in range(x_pert_curr.shape[0]):
                # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h - 1, w - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1

                n_squares = args.n_squares

                for i in range(n_squares):
                    center_h = np.random.randint(0, h - s)
                    center_w = np.random.randint(0, w - s)
                    x_curr_window = x_ori[i_img, :, center_h:center_h + s, center_w:center_w + s]
                    x_best_curr_window = x_pert_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
                    # prevent trying out a delta if it doesn't change x_ori (e.g. an overlapping patch)
                    while torch.sum(
                            torch.abs(torch.clamp(
                                x_curr_window + deltas_cur[i_img, :, center_h:center_h + s, center_w:center_w + s],
                                0, 1) - x_best_curr_window) < 10 ** -7) == c * s * s:
                        deltas_cur[i_img, :, center_h:center_h + s, center_w:center_w + s] = eps * random_sign(
                            size=[c, 1, 1]).to(device)
            x_new = torch.clamp(x_ori + deltas_cur, 0, 1)

            pred_s = predict_quality_score(x_new, args)
            loss_candidate = obj_function(pred_s, f_x)

            idx_improved = (loss_candidate < loss_min_curr).nonzero().squeeze().cpu()
            loss_mins[n][idx_improved] = loss_candidate[idx_improved].clone()
            x_pert_bests[n][idx_improved] = x_new[idx_improved].clone()
            pred_s_purt_cur_bests[n][idx_improved] = pred_s[idx_improved].cpu().numpy()

        srcc = scipy.stats.mstats.spearmanr(x=pred_s_purt_cur_bests, y=gt)[0]
        logger.log('srcc_law={:.4f}'.format(srcc))

        plcc = scipy.stats.mstats.pearsonr(x=pred_s_purt_cur_bests, y=gt)[0]
        logger.log('plcc_law={:.4f}'.format(plcc))

        RGO = []
        Dist = []
        for n in range(args.n_sample // args.bs):
            x_ori, f_x, pred_s_purt_cur_best = x_oris[n].clone(), fx[n], pred_s_purt_cur_bests[n]
            Dist.append(pred_s_purt_cur_best - f_x)
            for idx in range(x_ori.shape[0]):
                RGO.append((abs(pred_s_purt_cur_best[idx] - f_x[idx])) / max(10.0 - f_x[idx], f_x[idx] - 0.0))

        RGO = np.mean(RGO)

        # ['SRCC', 'RLCC', 'RGO', 'Deviation']
        loss = np.mean(sum(loss_mins, 0).cpu().numpy())
        Dist = sum(Dist, 0)
        div = np.mean(abs(Dist))
        list = [srcc, plcc, RGO, loss, div]
        stat = pd.DataFrame([list])
        stat.to_csv(stat_log_path, mode='a', header=False, index=False)
        dist = pd.DataFrame([Dist])
        dist.to_csv(dist_log_path, mode='a', header=False, index=False)

        adv_imgs = x_pert_bests
    return adv_imgs, x_oris

def seed_everything(seed=555):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='DBCNN')  # 'DBCNN' || 'UNIQUE' || 'TReS' || 'LIQE'
    parser.add_argument('--dataset', type=str, default='TID')  # 'LIVE' || 'CSIQ' || 'TID'
    parser.add_argument('--n_queries', type=int, default=100)
    parser.add_argument('--seed', type=int, default=555)
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--n_sample', type=int, default=50)
    parser.add_argument('--n_squares', type=int, default=2)
    parser.add_argument('--p_init', type=float, default=0.04)       # n_squares = 8, p_init = 0.09 on LIQE models;  n_squares = 2, p_init = 0.04 on other datasets
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--eps', type=float, default=3.0/255)
    parser.add_argument('--bs', type=int, default=1)

    args = parser.parse_args()


    seed_everything(seed=args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.model != 'TReS' and args.dataset == 'LIVE' or args.model == 'LIQE':
        args.bs = 1
    else:
        args.bs = 50

    param_run = 'model_{}_dataset_{}_nqueries_{:.0f}_seed_{:.0f}_p_init_{:0.2f}_nsquares_{:.0f}_eps_{:0.4f}'.format(
        args.model, args.dataset, args.n_queries, args.seed, args.p_init, args.n_squares, args.eps)
    log_path = '{}/log_run_{}_{}.txt'.format(args.log_dir, str(datetime.now())[:-7], param_run)
    logger = Logger(log_path)
    df = pd.DataFrame(columns=['SRCC', 'RLCC', 'RGO', 'Loss', 'Diviation'])
    stat_log_path = '{}/log_stat_{}_{}.csv'.format(args.log_dir, str(datetime.now())[:-7], param_run)
    df.to_csv(stat_log_path, index=False)
    dist_log_path = '{}/log_dist_{}_{}.csv'.format(args.log_dir, str(datetime.now())[:-7], param_run)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and dataset
    if args.model == 'DBCNN':
        model = torch.nn.DataParallel(DBCNN()).train(False).to(device)
        ckpt = './model_pt/DBCNN_' + args.dataset + '.pt'
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint)

    elif args.model == 'UNIQUE':
        model = BaseCNN()
        model = torch.nn.DataParallel(model).to(device)
        ckpt = './model_pt/UNIQUE.pt'
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint)

    elif args.model == 'TReS':
        model = TReS(device=device).to(device)
        ckpt = './model_pt/TReS_' + args.dataset + '.pt'
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint)

    elif args.model == 'LIQE':
        ckpt = './model_pt/LIQE.pt'
        model = LIQE(ckpt, device)

    else:
        print('Do not support.')
        exit(0)


    model.eval()

    if args.model == 'TReS':
        test_transform = transforms.Compose([
            transforms.CenterCrop(size=224),
            transforms.ToTensor()
        ])
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    dataset = ImageDataset(csv_file='./to_be_attacked/'+args.dataset + '_sample.txt', img_dir=dataset_path_dict[args.dataset],
                           transform=test_transform)
    dataset_loader = DataLoader(dataset, batch_size=args.bs)

    n_queries = np.ones(args.n_sample)
    fx = []
    gt = []
    with torch.no_grad():
        #######################################################
        # 0x00: compute the prediction score of the law image #
        #######################################################
        for i, (image, mos) in enumerate(dataset_loader):
            x = image.to(device)
            pred_score = predict_quality_score(x, args)
            mos = linear_rescale(mos, args)
            fx.append(pred_score.cpu().numpy())
            gt.append(mos.cpu().numpy())
        logger.log('fx = {}'.format(fx))
        srcc = scipy.stats.mstats.spearmanr(x=fx, y=gt)[0]
        logger.log('srcc_law={:.4f}'.format(srcc))

        plcc = scipy.stats.mstats.pearsonr(x=fx, y=gt)[0]
        logger.log('plcc_law={:.4f}'.format(plcc))

        dist = pd.DataFrame([sum(fx, 0)])
        dist.to_csv(dist_log_path, mode='a', header=False, index=False)

        adv_imgs, x_oris = attackIQA(dataset_loader=dataset_loader, args=args)

        pred_advs = []
        distortions = []
        RGO = []
        Dist = []

        for n in range(args.n_sample // args.bs):
            # a batch of images: x_oris[n]
            adv_img = adv_imgs[n].clone()
            x_ori = x_oris[n].clone()
            pred_adv = predict_quality_score(adv_img, args)
            pred_advs.append(pred_adv.cpu().numpy())
            distortions.append((abs((adv_img - x_ori).max(dim=1)[0].cpu().numpy())).max())

            fx.append(pred_score.cpu().numpy())

        srcc = scipy.stats.mstats.spearmanr(x=pred_advs, y=gt)[0]
        logger.log('srcc_law={:.4f}'.format(srcc))

        plcc = scipy.stats.mstats.pearsonr(x=pred_advs, y=gt)[0]
        logger.log('plcc_law={:.4f}'.format(plcc))

        lf = max(distortions)
        logger.log('linf_inifity_distortions={:.4f}'.format(lf))

        torch.save({'adv_images': adv_imgs},
                   args.save_dir + args.model + '_' + args.dataset + param_run + '_adv_imags.pth')

        logger.log('Done...')

