import torch 
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import numpy as np

from utils import init_dict, save_dict, curve_save, time_mark, print_cz, update_lr



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
    #print(N, C, D1, D2)
    temp_low_amp = np.zeros((C*D1, D2*N), dtype=float)
    temp_low_pha = np.zeros((C*D1, D2*N), dtype=float)
    for i in range(client_num):
        # N, C, D1, D2 = weights[i].size()
        #weights[i] = weights[i].cpu().numpy()
        if is_conv:
            weights[i] = weights[i].permute(1, 2, 3, 0).reshape((C*D1, D2*N))
        weights[i] = weights[i].cpu().numpy()

        client_fft = np.fft.fft2(weights[i], axes=(-2, -1))
        amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft)
        low_part_amp = np.fft.fftshift(amp_fft, axes=(-2, -1))
        temp_low_amp += low_part_amp
        low_part_pha = np.fft.fftshift(pha_fft, axes=(-2, -1))
        temp_low_pha += low_part_pha
    temp_low_amp = temp_low_amp / 4
    temp_low_pha = temp_low_pha / 4

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
        # get the mutated image
        fft_back_ = np.fft.ifft2(fft_back_, axes=(-2, -1))
        weights[i] = torch.FloatTensor(np.real(fft_back_))
        if is_conv:
            weights[i] = weights[i].reshape(C, D1, D2, N).permute(3, 0, 1, 2)
        #print(weights[i].shape)
    return weights


def avg_freq_all_finalFC(
    weights, 
    L=0.1, 
    ):
    client_num = len(weights)
    cls_num, D = weights[0].size()
    print('cls_num: {:d}, D: {:d}'.format(cls_num, D))
    # N = 1
    # C = 1
    # D1, D2 = weights[0].size() # 3*D
    #print(N, C, D1, D2)
    temp_low_amp = [np.zeros(D, dtype=float) for cls_idx in range(cls_num)]
    temp_low_pha = [np.zeros(D, dtype=float) for cls_idx in range(cls_num)]
    for i in range(client_num):
        weights[i] = weights[i].cpu().numpy()
        for cls_idx in range(cls_num):
            client_fft = np.fft.fft(weights[i][cls_idx, :], axis=-1)
            amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft)
            
            low_part_amp = np.fft.fftshift(amp_fft, axes=(-1,))
            temp_low_amp[cls_idx] += low_part_amp
            
            low_part_pha = np.fft.fftshift(pha_fft, axes=(-1,))
            temp_low_pha[cls_idx] += low_part_pha
    for cls_idx in range(cls_num):
        temp_low_amp[cls_idx] = temp_low_amp[cls_idx] / 4
        temp_low_pha[cls_idx] = temp_low_pha[cls_idx] / 4

    for i in range(client_num):
        for cls_idx in range(cls_num):
            client_fft = np.fft.fft(weights[i][cls_idx, :], axis=-1)
            amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft)
            low_part_amp = np.fft.fftshift(amp_fft, axes=(-1,))
            low_part_pha = np.fft.fftshift(pha_fft, axes=(-1,))

            h = low_part_amp.shape[0]
            b_h = (np.floor(h *L / 2)).astype(int)
            c_h = np.floor(h/2.0).astype(int)
            
            h1 = c_h-b_h
            h2 = c_h+b_h
            
            low_part_amp[h1:h2] = temp_low_amp[cls_idx][h1:h2]
            low_part_amp = np.fft.ifftshift(low_part_amp, axes=(-1,))

            low_part_pha[h1:h2] = temp_low_pha[cls_idx][h1:h2]
            low_part_pha = np.fft.ifftshift(low_part_pha, axes=(-1,))

            fft_back_ = low_part_amp * np.exp(1j * low_part_pha)
            # get the mutated image
            fft_back_ = np.fft.ifft(fft_back_, axis=-1)
            weights[i][cls_idx, :] = np.real(fft_back_)
            
            #print(weights[i].shape)
        #
        weights[i] = torch.FloatTensor(weights[i])
    return weights


def PFA(
    weights, 
    L,
    is_conv
    ):
    
    return avg_freq_all(weights=weights, L=L, is_conv=is_conv)
    
def PFA_finalFC(
    weights, 
    L
    ):
    
    return avg_freq_all_finalFC(weights=weights, L=L)



################# Key Function ########################
def communication(
    args, 
    server_model, 
    models, 
    original_models, 
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
                                            models[3].state_dict()[key].data
                                        ], 
                                        L=pfa_rate, 
                                        is_conv=True
                    )
                    for client_idx in range(client_num): # copy from server to each client
                        models[client_idx].state_dict()[key].data.copy_(temp_weights[client_idx])
                elif 'linear' in key and 'weight' in key and 'fc2' not in key:
                    temp_weights = PFA(
                                        [
                                            models[0].state_dict()[key].data,
                                            models[1].state_dict()[key].data,
                                            models[2].state_dict()[key].data,
                                            models[3].state_dict()[key].data
                                        ], 
                                        L=pfa_rate, 
                                        is_conv=False
                    )
                    for client_idx in range(client_num): # 
                        models[client_idx].state_dict()[key].data.copy_(temp_weights[client_idx])
                elif 'linear' in key and 'weight' in key and 'fc2' in key:
                    temp_weights = PFA_finalFC(
                                        [
                                            models[0].state_dict()[key].data,
                                            models[1].state_dict()[key].data,
                                            models[2].state_dict()[key].data,
                                            models[3].state_dict()[key].data
                                        ], 
                                        L=pfa_rate, 
                    )
                    for client_idx in range(client_num): # 
                        models[client_idx].state_dict()[key].data.copy_(temp_weights[client_idx])

                else:
                    print(key, '\t not bn, conv, fc layer, with param!')
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp) # non-bn layer，update the server model
                    for client_idx in range(client_num): # non-bn layer, from server to each client
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
 
    return server_model, models

