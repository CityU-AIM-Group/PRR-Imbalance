import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import numpy as np
from torch.autograd import Variable
from sklearn import metrics
from sklearn.metrics import roc_auc_score # 

from utils import print_cz



def DET(
    args, 
    model_deputy, 
    model, 
    train_loader, 
    optimizer_deputy, 
    optimizer, 
    loss_fun, 
    device, 
    DET_stage,
    local_proto,
    global_proto,
    logfile=None
    ):

    train_loss, train_acc, train_f1, train_multi_auc = test(args, model, train_loader, loss_fun, device, local_proto, global_proto)
    train_loss_deputy, train_acc_deputy, train_f1_deputy, train_multi_auc_deputy = test(args, model_deputy, train_loader, loss_fun, device, local_proto, global_proto)
    alpha1 = args.alpha1
    alpha2 = args.alpha2    
    # print_cz('-', f=logfile)
        
    if (train_f1_deputy < alpha1 * train_f1) or DET_stage == 0:
        DET_stage = 1
        print_cz('recover', f=logfile)
        # print_cz('personalized is teacher', f=logfile)
    elif (train_f1_deputy >= alpha1 * train_f1 and DET_stage == 1) or (DET_stage >= 2 and train_f1_deputy < alpha2 * train_f1):           
        DET_stage = 2
        print_cz('exchange', f=logfile)
        # print_cz('mutual learning', f=logfile)        
    elif train_f1_deputy >= alpha2 * train_f1 and DET_stage >= 2:          
        DET_stage = 3
        print_cz('sublimate', f=logfile)
        # print_cz('deputy is teacher', f=logfile)
    else:
        print_cz('***********************Logic error************************', f=logfile)
        DET_stage = 4
            
    model.train()
    model_deputy.train()
    model.to(device) # 
    model_deputy.to(device) # 
    num_data = 0
    correct = 0
    loss_all = 0
    loss_deputy_all = 0
    
    label_list_cz = [] # cz
    pred_list_cz = [] # cz
    output_list_cz = []

    proto_list = [[], [], []]
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad() # cz mark
        optimizer_deputy.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output,  _, feature = model(x)
        output_deputy,  _, feature_deputy = model_deputy(x)  

        # collect proto
        for i in range(x.shape[0]):
            if 'fc' in args.proto_type.lower():
                proto_list[y[i]].append(feature.clone().detach().cpu()[i].view(-1))
            elif 'logit' in args.proto_type.lower():
                proto_list[y[i]].append(output.clone().detach().cpu()[i].view(-1))
        #
        # if 'fc' in args.proto_type.lower():
        #     batch_protos = feature
        #     batch_protos_deputy = feature_deputy
        # elif 'logit' in args.proto_type.lower():
        #     batch_protos = output 
        #     batch_protos_deputy = output_deputy                  

        # cz 
        _, pred_cz = output.topk(1, 1, True, True)
        pred_list_cz.extend(
            ((pred_cz.cpu()).numpy()).tolist())
        label_list_cz.extend(
            ((y.cpu()).numpy()).tolist())

        if args.wk_iters < 2:
            # default mutual learning if wk_iters == 1
            loss_ce = loss_fun(output, y, local_proto=local_proto, global_proto=global_proto)
            loss_kl = F.kl_div(F.log_softmax(output, dim = 1), F.softmax(output_deputy.clone().detach(), dim=1), reduction='batchmean')
            loss = loss_ce + loss_kl
                
            loss_deputy_ce = loss_fun(output_deputy, y, local_proto=local_proto, global_proto=global_proto)           
            loss_deputy_kl = F.kl_div(F.log_softmax(output_deputy, dim = 1), F.softmax(output.clone().detach(), dim=1), reduction='batchmean')
            loss_deputy = loss_deputy_ce + loss_deputy_kl            
        else: # args.wk_iters>=2, default 5
            if DET_stage == 1:
                # personalized is teacher                
                loss_ce = loss_fun(output, y, local_proto=local_proto, global_proto=global_proto)
                loss = loss_ce
                
                loss_deputy_ce = loss_fun(output_deputy, y, local_proto=local_proto, global_proto=global_proto)
                loss_deputy_kl = F.kl_div(F.log_softmax(output_deputy, dim = 1), F.softmax(output.clone().detach(), dim=1), reduction='batchmean')
                loss_deputy = loss_deputy_ce + loss_deputy_kl

            elif DET_stage == 2:
                # mutual learning DET_stage = 2
                loss_ce = loss_fun(output, y, local_proto=local_proto, global_proto=global_proto)
                loss_kl = F.kl_div(F.log_softmax(output, dim = 1), F.softmax(output_deputy.clone().detach(), dim=1), reduction='batchmean')
                loss = loss_ce + loss_kl
                    
                loss_deputy_ce = loss_fun(output_deputy, y, local_proto=local_proto, global_proto=global_proto)           
                loss_deputy_kl = F.kl_div(F.log_softmax(output_deputy, dim = 1), F.softmax(output.clone().detach(), dim=1), reduction='batchmean')
                loss_deputy = loss_deputy_ce + loss_deputy_kl
                   
            elif DET_stage == 3:
                # deputy is teacher
                loss_ce = loss_fun(output, y, local_proto=local_proto, global_proto=global_proto)
                loss_kl = F.kl_div(F.log_softmax(output, dim = 1), F.softmax(output_deputy.clone().detach(), dim=1), reduction='batchmean')
                loss = loss_ce + loss_kl
                    
                loss_deputy_ce = loss_fun(output_deputy, y, local_proto=local_proto, global_proto=global_proto)           
                loss_deputy = loss_deputy_ce
                                          
            else:
                # default mutual learning
                loss_ce = loss_fun(output, y, local_proto=local_proto, global_proto=global_proto)
                loss_kl = F.kl_div(F.log_softmax(output, dim = 1), F.softmax(output_deputy.clone().detach(), dim=1), reduction='batchmean')
                loss = loss_ce + loss_kl
                    
                loss_deputy_ce = loss_fun(output_deputy, y, local_proto=local_proto, global_proto=global_proto)           
                loss_deputy_kl = F.kl_div(F.log_softmax(output_deputy, dim = 1), F.softmax(output.clone().detach(), dim=1), reduction='batchmean')
                loss_deputy = loss_deputy_ce + loss_deputy_kl
            output_list_cz.append(torch.nn.functional.softmax(output, dim=-1).cpu().detach().numpy())      
                     
        loss.backward()
        loss_deputy.backward()        
        loss_all += loss_ce.item()
        loss_deputy_all += loss_deputy_ce.item()                
        optimizer.step()
        optimizer_deputy.step()        
        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    
    # cz
    test_pred = np.concatenate(output_list_cz, axis=0)
    test_label = np.array(label_list_cz)
    mean_acc = 100*metrics.accuracy_score(label_list_cz, pred_list_cz)
    f1_macro = 100*metrics.f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    auc = 100.0*roc_auc_score(y_true=test_label, y_score=test_pred, multi_class='ovr')

    for j in range(len(proto_list)):
        proto_list[j] = torch.stack(proto_list[j], dim=0).mean(dim=0, keepdim=False)
    #
    proto_tensor = torch.stack(proto_list, dim=0).to(device) # C*D
    
    return DET_stage, loss_all/len(train_iter), loss_deputy_all/len(train_iter), mean_acc, f1_macro, auc, proto_tensor


def test(
    args,
    model, 
    test_loader, 
    loss_fun, 
    device,
    local_proto,
    global_proto
    ):
    model.eval()
    model.to(device) # 
    test_loss = 0
    correct = 0
    targets = []

    label_list_cz = [] # cz
    pred_list_cz = [] # cz
    output_list_cz = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output, _, feature = model(data)
         #
        # if 'fc' in args.proto_type.lower():
        #     batch_protos = feature
        # elif 'logit' in args.proto_type.lower():
        #     batch_protos = output 
        # cz 
        _, pred_cz = output.topk(1, 1, True, True)#
        pred_list_cz.extend(
            ((pred_cz.cpu()).numpy()).tolist())
        label_list_cz.extend(
            ((target.cpu()).numpy()).tolist())
        
        test_loss += loss_fun(output, target, local_proto=local_proto, global_proto=global_proto).item()
        pred = output.data.max(1)[1]
        output_list_cz.append(torch.nn.functional.softmax(output, dim=-1).cpu().detach().numpy())
        correct += pred.eq(target.view(-1)).sum().item()
    # cz
    test_pred = np.concatenate(output_list_cz, axis=0)
    test_label = np.array(label_list_cz)
    mean_acc = 100*metrics.accuracy_score(label_list_cz, pred_list_cz)
    f1_macro = 100*metrics.f1_score(y_true=label_list_cz, y_pred=pred_list_cz, average='macro')
    auc = 100.0*roc_auc_score(y_true=test_label, y_score=test_pred, multi_class='ovr')

    return test_loss/len(test_loader), mean_acc, f1_macro, auc

