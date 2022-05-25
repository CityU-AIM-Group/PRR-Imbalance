import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
print(os.getcwd())

import argparse
from numpy.lib.function_base import average
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image, ImageEnhance
import numpy as np
from utils.metric import Metrics, setup_seed
from utils.metric import evaluate
from models.deeplab import Deeplab
from models.deeplab_resnet import resnet18, resnet50, resnet101
import os.path as osp
import time
import warnings
import copy
import random
import math
from typing import Union
import matplotlib
import json
matplotlib.use('agg') 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


INPUT_SIZE = (256, 256)
seg_datasets = ['A', 'B', 'C', 'D', 'E']
client_num = len(seg_datasets)

global_category_info = [
    386166134, 7688842
]
client_category_info = [
    [82968948, 2260620],
    [139942506, 2499990],
    [46212370, 678638],
    [75166827, 1657749],
    [41875483, 591845]
]

def update_lr(lr, epoch, lr_step=20, lr_gamma=0.5):
    lr = lr * (lr_gamma ** (epoch // lr_step)) 
    return lr

def print_cz(str, f=None):
    if f is not None:
        print(str, file=f)
        if random.randint(0, 20) < 3:
            f.flush()
    print(str)

def init_dict(keys):
    d = {}
    for key in keys:
        d[key] = []
    return d

def save_dict(info_dict, theme, save_dir):
    
    with open(os.path.join(save_dir, 'infodict-{}.json'.format(theme)), 'w') as f:
        f.write(json.dumps(info_dict))

def time_mark():
    time_now = int(time.time())
    time_local = time.localtime(time_now)

    dt = time.strftime('%Y%m%d-%H%M%S', time_local)
    return(dt)

def DET(
    args, 
    model_agent, 
    model, 
    train_loader, 
    optimizer_agent, 
    optimizer, 
    loss_fun, 
    distill_loss_fun, 
    DET_stage,
    local_proto,
    global_proto,
    distill_weight,
    logfile=None
    ):

    metrics = Metrics(['recall', 'specificity', 'Dice',
                    'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])
    total_batch = math.ceil(len(train_loader.dataset) / args.batch_size)
    model.train()
    seg_loss = 0
    tic = time.time()

    train_loss, train_dice, train_acc, train_miou, train_recall = test(test_loader=train_loader, model=model, args=args, loss_fun=loss_fun, local_proto = local_proto, global_proto = global_proto)
    train_loss_agent, train_dice_agent, train_acc_agent, train_miou_agent, train_recall_agent = test(test_loader=train_loader, model=model_agent, args=args, loss_fun=loss_fun, local_proto = local_proto, global_proto = global_proto)
    alpha1 = args.alpha1
    alpha2 = args.alpha2    
    print_cz('-', f=logfile)
    print_cz('Difference value: {:.6f} in Dice'.format(train_dice_agent - train_dice), f=logfile)    
    
    if (train_dice_agent < alpha1 * train_dice) or DET_stage == 0:
        DET_stage = 1
        print_cz('client is teacher', f=logfile)
    elif (train_dice_agent >= alpha1 * train_dice and DET_stage == 1) or (DET_stage >= 2 and train_dice_agent < alpha2 * train_dice):           
        DET_stage = 2
        print_cz('mutuall learning', f=logfile)        
    elif train_dice_agent >= alpha2 * train_dice and DET_stage >= 2:          
        DET_stage = 3
        print_cz('agent is teacher', f=logfile)
    else:
        print_cz('***********************Logic error************************', f=logfile)
        DET_stage = 4

    model.train()
    model_agent.train()
    proto_list = [[], []]
    for i_iter, batch in enumerate(train_loader):

        data, name = batch
        image = data['image']
        label = data['label']
        image = Variable(image).cuda()
        label = Variable(label.long()).cuda()
        output, feature = model(image)
        output_agent, feature_agent = model_agent(image)

        scale = feature.size()[2] / label.size()[2]
        Upsample = torch.nn.UpsamplingNearest2d(scale_factor=scale)
        label_p = Upsample(label.float().unsqueeze(1))

        # collect class-wise features for further prototype computation
        for i in range(image.shape[0]):
            proto_list[0].append(torch.mean(feature.mul(1-label_p), dim=(2,3)).clone().detach().cpu()[i].view(-1))
            proto_list[1].append(torch.mean(feature.mul(label_p), dim=(2,3)).clone().detach().cpu()[i].view(-1))

        if args.wk_iters < 2:
            #mutuall learning if wk_iters ==1
            loss_ce = loss_fun(output, label, local_proto=local_proto, global_proto=global_proto)
            loss_distill = distill_loss_fun(F.softmax(output, dim = 1), F.softmax(output_agent, dim=1).clone().detach())
            loss = loss_ce + distill_weight * loss_distill  
            loss_agent_ce = loss_fun(output_agent, label, local_proto=local_proto, global_proto=global_proto)           
            loss_agent_distill = distill_loss_fun(F.softmax(output_agent, dim = 1), F.softmax(output, dim=1).clone().detach())
            loss_agent = loss_agent_ce + distill_weight * loss_agent_distill            
        else:
            if DET_stage == 1:
                # client is teacher                
                loss_ce = loss_fun(output, label, local_proto=local_proto, global_proto=global_proto)
                loss = loss_ce
                loss_agent_ce = loss_fun(output_agent, label, local_proto=local_proto, global_proto=global_proto)
                loss_agent_distill = distill_loss_fun(F.softmax(output_agent, dim = 1), F.softmax(output, dim=1).clone().detach())
                loss_agent = loss_agent_ce + distill_weight * loss_agent_distill

            elif DET_stage == 2:
                #mutuall learning DET_stage = 2
                loss_ce = loss_fun(output, label, local_proto=local_proto, global_proto=global_proto)
                loss_distill = distill_loss_fun(F.softmax(output, dim = 1), F.softmax(output_agent, dim=1).clone().detach())
                loss = loss_ce + distill_weight * loss_distill
                loss_agent_ce = loss_fun(output_agent, label, local_proto=local_proto, global_proto=global_proto)           
                loss_agent_distill = distill_loss_fun(F.softmax(output_agent, dim = 1), F.softmax(output, dim=1).clone().detach())
                loss_agent = loss_agent_ce + distill_weight * loss_agent_distill
                   
            elif DET_stage ==3:
                 #agent is teacher
                loss_ce = loss_fun(output, label, local_proto=local_proto, global_proto=global_proto)
                loss_distill = distill_loss_fun(F.softmax(output, dim = 1), F.softmax(output_agent, dim=1).clone().detach())
                loss = loss_ce + distill_weight * loss_distill
                loss_agent_ce = loss_fun(output_agent, label, local_proto=local_proto, global_proto=global_proto)           
                loss_agent = loss_agent_ce
                                          
            else:
                # mutuall learning
                loss_ce = loss_fun(output, label, local_proto=local_proto, global_proto=global_proto)
                loss_distill = distill_loss_fun(F.softmax(output, dim = 1), F.softmax(output_agent, dim=1).clone().detach())
                loss = loss_ce + distill_weight * loss_distill
                    
                loss_agent_ce = loss_fun(output_agent, label, local_proto=local_proto, global_proto=global_proto)          
                loss_agent_distill = distill_loss_fun(F.softmax(output_agent, dim = 1), F.softmax(output, dim=1).clone().detach())
                loss_agent = loss_agent_ce + distill_weight * loss_agent_distill
                                
        loss.backward(retain_graph=True)
        loss_agent.backward()                    
        optimizer.step()
        optimizer_agent.step()  
        optimizer.zero_grad()
        optimizer_agent.zero_grad()

        seg_loss += loss.item()
        _recall, _specificity, _Dice, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, label)
        metrics.update(recall=_recall, specificity=_specificity,
                           Dice=_Dice, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                           IoU_bg=_IoU_bg, IoU_mean=_IoU_mean
                           )
    metrics_result = metrics.mean(total_batch)

    for j in range(len(proto_list)):
        proto_list[j] = torch.stack(proto_list[j], dim=0).mean(dim=0, keepdim=False)
    proto_tensor = torch.stack(proto_list, dim=0).cuda() # C*D

    return DET_stage, seg_loss, metrics_result['Dice'] * 100, metrics_result['ACC_overall'] * 100, metrics_result['IoU_mean'] * 100, metrics_result['recall'] * 100, proto_tensor


def test(
    test_loader, 
    model, 
    args,  
    loss_fun,
    local_proto,
    global_proto):
    metrics = Metrics(['recall', 'specificity', 'Dice',
                    'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])
    total_batch = len(test_loader.dataset)
    model.eval()
    seg_loss = 0

    with torch.no_grad(): #########
        for i_iter, batch in enumerate(test_loader):
            data, name = batch
            image = data['image']
            label = data['label']
            image = Variable(image).cuda()
            label = Variable(label.long()).cuda()
            output, feature = model(image)
            
            loss_source = loss_fun(output, label, local_proto=local_proto, global_proto=global_proto)
            seg_loss += loss_source.item()

            _recall, _specificity, _Dice, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, label)
            metrics.update(recall=_recall, specificity=_specificity,
                            Dice=_Dice, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                            IoU_bg=_IoU_bg, IoU_mean=_IoU_mean
                            )
        metrics_result = metrics.mean(total_batch)

    return seg_loss, metrics_result['Dice'] * 100, metrics_result['ACC_overall'] * 100, metrics_result['IoU_mean'] * 100, metrics_result['recall'] * 100


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lr_step", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default='fedtmi',
                    help="federated learning method.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", type=bool, default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--wk_iters", type = int, default=5, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')  
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument('--save_path', type = str, default='./experiment/')
    parser.add_argument('--seed', type = int, default=1, help = 'seed')
    parser.add_argument('--weight', type = bool, default=True, help='class imbalance weight')
    parser.add_argument("--wd", type=float, default=1e-6)

    # PFA
    parser.add_argument("--l_rate", type=float, default=0.7)
    parser.add_argument('--pfa_mode', type = str, default='avg_freq_all', help='avg_freq_amp | avg_freq_pha | avg_freq_all')
    # DET
    parser.add_argument('--alpha1', type = float, default= 0.7, help = 'alpha1')
    parser.add_argument('--alpha2', type = float, default= 0.9, help = 'alpha2')
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument('--network', type = str, default='Res18')
    parser.add_argument('--distill_weight', type = int, default=10)

    parser.add_argument('--loss_beta', type = float, default= 0.8)
    parser.add_argument('--clamp_thres', type = float, default= 0, help = 'clamp thres for proto factor, as 0 or 1')
    parser.add_argument('--init_iter', type = int, default= 1, help = 'proto-based loss after the init_iter')
    parser.add_argument('--proto_type', type = str, default='fc', help='fc | logit')
    parser.add_argument('--global_proto_type', type = str, default='avg', help='avg | gaussian')
    parser.add_argument('--tau', type = float, default= 1, help = 'adjust for cosine score')
    #
    parser.add_argument('--theme', type = str, default='')
    return parser.parse_args()

args = get_arguments()

def main():
    setup_seed(20)
    seed= args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    if args.pretrain:
        model_init = 'pretrain'
    else:
        model_init = 'scratch'
    
    log_path = args.save_path + time_mark() + '_' + args.theme + '_' + args.mode + '_' +args.optim +'_{}'.format(model_init) + '_lr' + str(args.lr) + '_step'+str(args.lr_step) + '_seed'+str(args.seed) + '_wd'+str(args.wd) +'_iters'+str(args.iters)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
    print_cz(os.getcwd(), f=logfile)

    print_cz('==={}==='.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), f=logfile)
    print_cz('===Setting===', f=logfile)
    print_cz('    mode: {}'.format(args.mode), f=logfile)
    print_cz('    optim: {}'.format(args.optim), f=logfile)
    print_cz('    lr: {}'.format(args.lr), f=logfile)
    print_cz('    lr_step: {}'.format(args.lr_step), f=logfile)
    print_cz('    iters: {}'.format(args.iters), f=logfile)
    print_cz('    wk_iters: {}'.format(args.wk_iters), f=logfile)
    print_cz('    weight: {}'.format(args.weight), f=logfile)
    print_cz('    distill_weight: {}'.format(args.distill_weight), f=logfile)
    #
    print_cz('    lr_step: {}'.format(args.lr_step), f=logfile)
    print_cz('    alpha1: {}'.format(args.alpha1), f=logfile)
    print_cz('    alpha2: {}'.format(args.alpha2), f=logfile)
    print_cz('    network: {}'.format(args.network), f=logfile)
    if args.pretrain:
        print_cz('    pretrain: {}'.format('True'), f=logfile)
    else:
        print_cz('    pretrain: {}'.format('False'), f=logfile)

    train_loaders, test_loaders, valid_loaders = prepare_data_fed_aug(
        batch_size=args.batch_size, 
        data_dir='./data'
        )

    train_len = [len(loader) for loader in train_loaders]
    test_len  = [len(loader) for loader in test_loaders]
    print_cz('Train loader len:  {}'.format(train_len), f=logfile)
    print_cz('Test  loader len:  {}'.format(test_len), f=logfile)

    ############ record curve #####################
    info_keys = ['test_epochs', 'test_dice', 'test_miou']
    info_dicts = {
        'A': init_dict(keys=info_keys), 
        'B': init_dict(keys=info_keys), 
        'C': init_dict(keys=info_keys), 
        'D': init_dict(keys=info_keys),
        'E': init_dict(keys=info_keys),
        'Average': init_dict(keys=info_keys)}

    loss_fun_init = CPA_Loss_init(
    class_counts=global_category_info,
    beta=args.loss_beta
    )
    loss_fun_refine = CPA_Loss(
        class_counts=global_category_info,
        beta=args.loss_beta,
        clamp_thres=args.clamp_thres,
        tau=args.tau
    )
    distill_loss_fun = nn.MSELoss(reduction='mean')
    
    if args.network == 'Res18':
        server_model = resnet18(pretrained = args.pretrain, proto=True).cuda()
    elif args.network == 'Res50':
        server_model = resnet50(pretrained = args.pretrain, proto=True).cuda()
    elif args.network == 'Res101':
        server_model = resnet101(pretrained = args.pretrain, proto=True).cuda()

    client_weights = [1/client_num for i in range(client_num)]
    models_single = [copy.deepcopy(server_model).cuda() for idx in range(client_num)]
    models_agent = [copy.deepcopy(server_model).cuda() for idx in range(client_num)]

    local_protos = torch.ones(5, 2, 512).cuda()
    global_proto = global_gaussian_proto(local_protos)


    start_time = time.time()
    for a_iter in range(args.iters):

        if a_iter>=args.init_iter:
            source_criterion = loss_fun_refine
            print_cz('loss_fun_refine, FederatedImbalanceLoss', f=logfile)
        else:
            source_criterion = loss_fun_init
            print_cz('loss_fun_init, CPA_Loss_init', f=logfile)

        iter_start_time = time.time()
        lr_current = update_lr(lr=args.lr, epoch=a_iter, lr_step=args.lr_step, lr_gamma=args.lr_gamma)
        if (args.optim).lower() == 'sgd':
            optimizers = [optim.SGD(params=models_single[idx].parameters(), lr=lr_current, weight_decay=args.wd) for idx in range(client_num)]
            optimizers_agent = [optim.SGD(params=models_agent[idx].parameters(), lr=lr_current, weight_decay=args.wd) for idx in range(client_num)]
        elif (args.optim).lower() == 'adam':
            optimizers = [optim.Adam(params=models_single[idx].parameters(), lr=lr_current, weight_decay=args.wd) for idx in range(client_num)]
            optimizers_agent = [optim.Adam(params=models_agent[idx].parameters(), lr=lr_current, weight_decay=args.wd) for idx in range(client_num)]
        
        #
        DET_stages = [0 for i in range(client_num)] # DET status initialization

        for wi in range(args.wk_iters):
            local_protos_list = []
            print_cz("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters), f=logfile)
            print_cz("=== lr_current:  {:.4e} ===".format(lr_current), f=logfile)
            for client_idx in range(client_num):
                DET_stages[client_idx], train_loss_, train_dice_, train_acc_, train_miou_, train_recall_, local_proto = DET(
                    args=args, 
                    model_agent=models_agent[client_idx],
                    model=models_single[client_idx],
                    train_loader=train_loaders[client_idx],
                    optimizer_agent=optimizers_agent[client_idx],
                    optimizer=optimizers[client_idx],
                    loss_fun=source_criterion,
                    distill_loss_fun=distill_loss_fun,
                    DET_stage=DET_stages[client_idx],
                    local_proto=local_protos[client_idx], 
                    global_proto=global_proto,
                    distill_weight=args.distill_weight,
                    logfile=logfile)
                print_cz(' {:<5s}| Train_Loss: {:.2f} | Dice: {:.2f}  Acc: {:.2f}  mIoU: {:.2f}  Recall: {:.2f}'.format(seg_datasets[client_idx] ,train_loss_, train_dice_, train_acc_, train_miou_, train_recall_), f=logfile)

                local_protos_list.append(local_proto) #
      
            test_average = []
            for client_idx in range(client_num):
                test_loss_, test_dice_, test_acc_, test_miou_, test_recall_ = test(test_loaders[client_idx], models_single[client_idx], args, source_criterion, local_protos[client_idx], global_proto)

                test_average.append([test_loss_, test_dice_, test_acc_, test_miou_, test_recall_])         
                print_cz(' {:<11s}| Test  Loss: {:.2f} | Dice: {:.2f}  Acc: {:.2f}  mIoU: {:.2f}  Recall: {:.2f}'.format(seg_datasets[client_idx], test_loss_, test_dice_, test_acc_, test_miou_, test_recall_), f=logfile)
            test_mean = np.mean(np.array(test_average), axis=0)
            print_cz(' {:<11s}| Test  Loss: {:.2f} | Dice: {:.2f}  Acc: {:.2f}  mIoU: {:.2f}  Recall: {:.2f}'.format('Average', test_mean[0], test_mean[1], test_mean[2], test_mean[3], test_mean[4]), f=logfile)
            
            local_protos = torch.stack(
                    local_protos_list,
                    dim=0
                    ) 
            if 'avg' in args.global_proto_type.lower():
                global_proto = global_avg_proto(local_protos) ## 
            else:
                global_proto = global_gaussian_proto(local_protos)

            for client_idx in range(client_num):
                cosine_score = proto_factor_cosine(local_protos[client_idx], global_proto)
                proto_factor = tau_func(cosine_score, args.tau)
        if args.mode.lower() != 'singleset' and args.mode.lower() != 'central':
            print("Aggragation by {}".format(args.mode.lower()))
            server_model, models_agent = communication(
                args=args, 
                server_model=server_model, 
                models=models_agent, 
                client_weights=client_weights, 
                a_iter=a_iter)
        
        test_average = []
        for client_idx in range(client_num):
            test_loss_, test_dice_, test_acc_, test_miou_, test_recall_ = test(test_loaders[client_idx], models_single[client_idx], args, source_criterion, local_protos[client_idx], global_proto)

            test_average.append([test_loss_, test_dice_, test_acc_, test_miou_, test_recall_])
            print_cz(' {:<11s}| Test  Loss: {:.2f} | Dice: {:.2f}  Acc: {:.2f}  mIoU: {:.2f}  Recall: {:.2f}'.format(seg_datasets[client_idx], test_loss_, test_dice_, test_acc_, test_miou_, test_recall_), f=logfile)
        test_mean = np.mean(np.array(test_average), axis=0)
        print_cz(' {:<11s}| Test  Loss: {:.2f} | Dice: {:.2f}  Acc: {:.2f}  mIoU: {:.2f}  Recall: {:.2f}'.format('Average', test_mean[0], test_mean[1], test_mean[2], test_mean[3], test_mean[4]), f=logfile)
        
        print_cz(' Iter time:  {:.1f} min'.format((time.time()-iter_start_time)/60.0), f=logfile)
    print_cz(' Total time:  {:.2f} h'.format((time.time()-start_time)/3600.0), f=logfile)

def avg_freq_all(
    weights, 
    L=0.1, 
    is_conv=True
    ):
    client_num = len(weights)
    
    if is_conv:
        N, C, D1, D2 = weights[0].size()
    else:
        N = 1
        C = 1
        D1, D2 = weights[0].size()
    temp_low_amp = np.zeros((C*D1, D2*N), dtype=float)
    temp_low_pha = np.zeros((C*D1, D2*N), dtype=float)
    for i in range(client_num):
        if is_conv:
            weights[i] = weights[i].permute(1, 2, 3, 0).reshape((C*D1, D2*N))
        weights[i] = weights[i].cpu().numpy()
        client_fft = np.fft.fft2(weights[i], axes=(-2, -1))
        amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft)
        low_part_amp = np.fft.fftshift(amp_fft, axes=(-2, -1))
        temp_low_amp += low_part_amp
        low_part_pha = np.fft.fftshift(pha_fft, axes=(-2, -1))
        temp_low_pha += low_part_pha
    temp_low_amp = temp_low_amp / 5
    temp_low_pha = temp_low_pha / 5
    for i in range(client_num):
        client_fft = np.fft.fft2(weights[i], axes=(-2, -1))
        amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft)
        low_part_amp = np.fft.fftshift(amp_fft, axes=(-2, -1))
        low_part_pha = np.fft.fftshift(pha_fft, axes=(-2, -1))

        h, w = low_part_amp.shape
        b_h = (np.floor(h *L / 2)).astype(int)
        b_w = (np.floor(w *L / 2)).astype(int)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)

        h1 = c_h-b_h
        h2 = c_h+b_h
        w1 = c_w-b_w
        w2 = c_w+b_w
        low_part_amp[h1:h2,w1:w2] = temp_low_amp[h1:h2,w1:w2]
        low_part_amp = np.fft.ifftshift(low_part_amp, axes=(-2, -1))
        low_part_pha[h1:h2,w1:w2] = temp_low_pha[h1:h2,w1:w2]
        low_part_pha = np.fft.ifftshift(low_part_pha, axes=(-2, -1))
        fft_back_ = low_part_amp * np.exp(1j * low_part_pha)
        #
        fft_back_ = np.fft.ifft2(fft_back_, axes=(-2, -1))
        weights[i] = torch.FloatTensor(np.real(fft_back_))
        if is_conv:
            weights[i] = weights[i].reshape(C, D1, D2, N).permute(3, 0, 1, 2)
        #print(weights[i].shape)
    return weights

def PFA(
    weights, 
    L,
    is_conv
    ):
    return avg_freq_all(weights=weights, L=L, is_conv=is_conv)

def communication(
    args, 
    server_model, 
    models, 
    client_weights, 
    a_iter
    ):
    pfa_rate = args.l_rate + (a_iter / args.iters) * (0.95 - args.l_rate)
    client_num = len(client_weights) # 
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            if 'bn' not in key: #not bn
                if 'conv' in key and 'weight' in key:
                    temp_weights = PFA(
                                    [
                                        models[0].state_dict()[key].data,
                                        models[1].state_dict()[key].data,
                                        models[2].state_dict()[key].data,
                                        models[3].state_dict()[key].data,
                                        models[4].state_dict()[key].data
                                    ], 
                                    L=pfa_rate, 
                                    is_conv=True
                    )
                    for client_idx in range(client_num): # copy from server to each client
                        models[client_idx].state_dict()[key].data.copy_(temp_weights[client_idx])
                elif 'linear' in key and 'weight' in key:
                    temp_weights = PFA(
                                    [
                                        models[0].state_dict()[key].data,
                                        models[1].state_dict()[key].data,
                                        models[2].state_dict()[key].data,
                                        models[3].state_dict()[key].data,
                                        models[4].state_dict()[key].data
                                    ], 
                                    L=pfa_rate, 
                                    is_conv=False
                    )
                    for client_idx in range(client_num): # 
                        models[client_idx].state_dict()[key].data.copy_(temp_weights[client_idx])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp) # non-bn layer，update the server model
                    for client_idx in range(client_num): # non-bn layer, from server to each client
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models


class CPA_Loss_init(nn.Module):
    """
    Args:
    class_counts: The list of the number of samples for each class. 
    beta: Scale parameter to adjust the strength.
    """

    def __init__(self, class_counts: Union[list, np.array], beta: float = 0.8):
        super(CPA_Loss_init, self).__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** beta
        # print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits, targets, **kwargs):
        dice_loss = DiceLoss()(torch.softmax(logits, dim=1)[:, 1, :, :].unsqueeze(1), targets)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, self.num_labels)
        targets = F.one_hot(targets, self.num_labels).reshape(-1, self.num_labels)
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)
        sigma = numerator / (denominator + self.eps)
        sigma2 = numerator / (self.eps + torch.exp(logits)[:, None, :].sum(axis=-1))
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)*0.5 + (- targets * torch.log(sigma2 + self.eps)).sum(-1)*0.5
        return loss.mean() + dice_loss

class CPA_Loss(nn.Module):
    """
    Args:
    class_counts: The list of the number of samples for each class. 
    beta: Scale parameter to adjust the strength.
    """
    def __init__(
        self, 
        class_counts: Union[list, np.array], 
        beta: float = 0.8,
        clamp_thres: float = 0,
        tau: float = 3.0
        ):
        super(CPA_Loss, self).__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** beta
        # print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.global_factor = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6
        self.clamp_thres = clamp_thres
        self.tau = tau
    
    def proto_factor_cosine(self, source_proto, target_proto):
        """
        [C, D]: D is 64 or 4
        """
        # factor = 1
        norm_source = torch.norm(source_proto, dim=-1, keepdim=False)
        norm_target = torch.norm(target_proto.detach(), dim=-1, keepdim=False) # [C]
        factor_refined = torch.sum(source_proto*target_proto.detach(), dim=-1, keepdim=False)/(norm_source*norm_target+self.eps)
        return factor_refined # [C]
    
    def forward(self, logits, targets, local_proto, global_proto):
        dice_loss = DiceLoss()(torch.softmax(logits, dim=1)[:, 1, :, :].unsqueeze(1), targets)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, self.num_labels)
        targets = F.one_hot(targets, self.num_labels).reshape(-1, self.num_labels)
        self.global_factor = self.global_factor.to(targets.device) # [C, C]
        max_element, _ = logits.max(axis=-1)
        # [N, C]
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits) # [N, C]
        denominator = (
            (1 - targets)[:, None, :]
            * self.global_factor[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits) # [N, C]

        sigma = numerator / (denominator + self.eps) # [N, C]
        sigma2 = numerator / (self.eps + torch.exp(logits)[:, None, :].sum(axis=-1))
        # proto factor
        cosine_score = self.proto_factor_cosine(source_proto=local_proto, target_proto=global_proto)
        proto_factor = (1+self.tau)/(cosine_score+self.tau) #
        # sum in categories
        loss = (- proto_factor.view(1, -1) * targets * torch.log(sigma + self.eps)).sum(-1)*0.5 + (- targets * torch.log(sigma2 + self.eps)).sum(-1)*0.5 # [N]
        return loss.mean() + dice_loss 

def global_avg_proto(local_protos):
    # local_protos: client_num*C*D
    return torch.mean(local_protos, dim=0, keepdim=False) # C*D

def global_gaussian_proto(local_protos):
    # local_protos: client_num*C*D
    mean = torch.mean(local_protos, dim=0, keepdim=False)
    std = torch.clamp(
        torch.std(local_protos, dim=0, keepdim=False),
        min=1
        )
    sample = torch.randn(mean.shape).to(mean.device)
    return sample * std + mean # C*D

def proto_factor_cosine(local_proto, global_proto):
    """
    [C, D]: D is 64 or 4
    """
    # factor = 1
    norm_local = torch.norm(local_proto, dim=-1, keepdim=False)
    norm_global = torch.norm(global_proto, dim=-1, keepdim=False) # [C]
    factor_refined = torch.sum(local_proto*global_proto, dim=-1, keepdim=False)/(norm_local*norm_global+1e-6)
    return factor_refined # [C]

def tau_func(cosine_score, tau):
    proto_factor = (1+tau)/(cosine_score+tau) #
    return proto_factor

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, weight=None):
        smooth = 1
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        if weight is not None:
            dice_score = weight * dice_score
            dice_loss = weight.sum() /size - dice_score.sum()/size
        else:
            dice_loss = 1 - dice_score.sum()/size
        return dice_loss

def randomRotation(image, label):
    random_angle = np.random.randint(1, 60)
    return image.rotate(random_angle, Image.BICUBIC), label.rotate(random_angle, Image.NEAREST)

def randomColor(image):
    random_factor = np.random.randint(0, 31) / 10.  
    color_image = ImageEnhance.Color(image).enhance(random_factor)  
    random_factor = np.random.randint(10, 21) / 10.  
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
    return ImageEnhance.Sharpness(brightness_image).enhance(random_factor)  

# Prostate Dataset
class Prostate(Dataset):
    def __init__(self, root, data_dir, mode='train', is_mirror=True, is_pseudo=None, max_iter=None):
        self.root = root
        self.data_dir = data_dir
        self.is_pseudo = is_pseudo
        self.is_mirror = is_mirror
        self.mode = mode
        self.imglist = []
        self.gtlist = []

        self.img_ids = [i_id.strip() for i_id in open(self.data_dir)]
        if not max_iter == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iter) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:

            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name
            })

    def __getitem__(self, index):
        img_path = self.files[index]["image"]
        gt_path = self.files[index]["label"]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256), Image.BICUBIC)
        gt = Image.open(gt_path).convert('L')
        gt = gt.resize((256, 256), Image.NEAREST)
        if self.mode == 'train':
            img, gt = randomRotation(img, gt)
            if self.is_mirror:
                flip = np.random.choice(2) * 2 - 1
                img = img[:, :, ::flip]
                gt = gt[:, ::flip]

        img = np.asarray(img, np.float32)
        img = img / 255
        gt = np.asarray(gt, np.float32)
        gt = gt / 255
        img = img[:, :, ::-1]  # change to BGR
        img = img.transpose((2, 0, 1))

        data = {'image': img.copy(), 'label': gt.copy()}
        return data, gt_path

    def __len__(self):
        return len(self.files)

def prepare_data_fed_aug(
    batch_size=16, 
    data_dir='./data'
    ):
    np.random.seed(22)
    # A
    trainset_A     = Prostate(root=data_dir+'/A', data_dir=data_dir+'/A/train.lst', mode='Train')
    testset_A     = Prostate(root=data_dir+'/A', data_dir=data_dir+'/A/test.lst', mode='Test')
    validset_A     = Prostate(root=data_dir+'/A', data_dir=data_dir+'/A/valid.lst', mode='Valid')
    # B
    trainset_B     = Prostate(root=data_dir+'/B', data_dir=data_dir+'/B/train.lst', mode='Train')
    testset_B     = Prostate(root=data_dir+'/B', data_dir=data_dir+'/B/test.lst', mode='Test')
    validset_B     = Prostate(root=data_dir+'/B', data_dir=data_dir+'/B/valid.lst', mode='Valid')
    # C
    trainset_C     = Prostate(root=data_dir+'/C', data_dir=data_dir+'/C/train.lst', mode='Train')
    testset_C     = Prostate(root=data_dir+'/C', data_dir=data_dir+'/C/test.lst', mode='Test')
    validset_C     = Prostate(root=data_dir+'/C', data_dir=data_dir+'/C/valid.lst', mode='Valid')
    # D
    trainset_D     = Prostate(root=data_dir+'/D', data_dir=data_dir+'/D/train.lst', mode='Train')
    testset_D     = Prostate(root=data_dir+'/D', data_dir=data_dir+'/D/test.lst', mode='Test')
    validset_D     = Prostate(root=data_dir+'/D', data_dir=data_dir+'/D/valid.lst', mode='Valid')
    # E
    trainset_E     = Prostate(root=data_dir+'/E', data_dir=data_dir+'/E/train.lst', mode='Train')
    testset_E     = Prostate(root=data_dir+'/E', data_dir=data_dir+'/E/test.lst', mode='Test')
    validset_E     = Prostate(root=data_dir+'/E', data_dir=data_dir+'/E/valid.lst', mode='Valid')
    
    train_loader_A = torch.utils.data.DataLoader(trainset_A, batch_size=batch_size,  shuffle=True)
    test_loader_A  = torch.utils.data.DataLoader(testset_A, batch_size=1, shuffle=False)
    valid_loader_A  = torch.utils.data.DataLoader(validset_A, batch_size=1, shuffle=False)

    train_loader_B = torch.utils.data.DataLoader(trainset_B, batch_size=batch_size,  shuffle=True)
    test_loader_B = torch.utils.data.DataLoader(testset_B, batch_size=1, shuffle=False)
    valid_loader_B  = torch.utils.data.DataLoader(validset_B, batch_size=1, shuffle=False)

    train_loader_C = torch.utils.data.DataLoader(trainset_C, batch_size=batch_size,  shuffle=True)
    test_loader_C = torch.utils.data.DataLoader(testset_C, batch_size=1, shuffle=False)
    valid_loader_C  = torch.utils.data.DataLoader(validset_C, batch_size=1, shuffle=False)

    train_loader_D = torch.utils.data.DataLoader(trainset_D, batch_size=batch_size,  shuffle=True)
    test_loader_D = torch.utils.data.DataLoader(testset_D, batch_size=1, shuffle=False)
    valid_loader_D  = torch.utils.data.DataLoader(validset_D, batch_size=1, shuffle=False)

    train_loader_E = torch.utils.data.DataLoader(trainset_E, batch_size=batch_size,  shuffle=True)
    test_loader_E = torch.utils.data.DataLoader(testset_E, batch_size=1, shuffle=False)
    valid_loader_E  = torch.utils.data.DataLoader(validset_E, batch_size=1, shuffle=False)

    train_loaders = [train_loader_A, train_loader_B, train_loader_C, train_loader_D, train_loader_E]
    test_loaders  = [test_loader_A, test_loader_B, test_loader_C, test_loader_D, test_loader_E]
    valid_loaders  = [valid_loader_A, valid_loader_B, valid_loader_C, valid_loader_D, valid_loader_E]

    return train_loaders, test_loaders, valid_loaders


if __name__ == '__main__':
    
    main()
