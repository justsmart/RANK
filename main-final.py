import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model_wf import get_model
import evaluation
import torch
import numpy as np
from myloss import Loss
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
def train(loader, model, loss_model, opt, sche, epoch, dep_graph,logger):
    assert torch.sum(torch.isnan(dep_graph)).item() == 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    global all_z,all_label,all_inc_V,all_inc_L
    # all_z = torch.tensor([]).to('cuda:0')
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        label = label.to('cuda:0')
        # print(label.shape,label.sum())
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        x_bar_list, target_pre, fusion_z, individual_zs, individual_preds, dis_score = model(data,inc_V_ind)
        z_nvd = torch.stack(individual_zs,dim=1)
        if epoch==0:
            all_z = torch.cat((all_z,z_nvd.clone().detach()),dim=0)
        else:
            all_z[i*label.size(0):(i+1)*label.size(0),:,:]=z_nvd.clone().detach()
        #######1
        tru_score = torch.abs((label.unsqueeze(1).mul(torch.log(individual_preds + 1e-10)) \
                                + (1-label.unsqueeze(1)).mul(torch.log(1 - individual_preds + 1e-10))).mul(inc_L_ind.unsqueeze(1))).sum(-1)/inc_L_ind.unsqueeze(1).sum(-1)
        tru_score[(1-inc_V_ind).bool()] = 1e9
        tru_score = F.softmax(-tru_score,dim=-1)
    
        loss_dis = torch.abs(tru_score.mul(torch.log(dis_score + 1e-10)).sum())/(dis_score.shape[0])
        assert torch.sum(torch.isnan(loss_dis)).item() == 0

        loss_Gp = loss_model.grather_positive_pairs(individual_zs,inc_V_ind)

        loss_CL = loss_model.New_CE9(target_pre,label,inc_L_ind,dep_graph)

        assert torch.sum(torch.isnan(loss_CL)).item() == 0
        if epoch>0:
            loss_LG = loss_model.all_label_guide(z_nvd, label, inc_V_ind, inc_L_ind, all_z, all_label, all_inc_V, all_inc_L, i, args.batch_size)
        else: loss_LG = 0
        loss_AE = 0
        for iv, x_bar in enumerate(x_bar_list):
            loss_AE += loss_model.wmse_loss(x_bar, data[iv], inc_V_ind[:, iv])
        loss = loss_CL + args.gamma * loss_AE  + loss_Gp * (1-args.beta**epoch) + args.alpha * loss_LG + loss_dis

        if epoch ==0:
            all_label = torch.cat((all_label,label),dim=0) 
            all_inc_V = torch.cat((all_inc_V,inc_V_ind),dim=0)
            all_inc_L = torch.cat((all_inc_L,inc_L_ind),dim=0)
        
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        
        opt.step()

        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()

    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    # print("all0",all0)
    return losses,model

def test(loader, model, loss_model, epoch, dep_graph,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        # data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        x_bar_list, pred, fusion_z, individual_zs, _, _ = model(data,inc_V_ind)
        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
        

        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        ap=evaluation_results[0], 
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results


def main(args,file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' + 
                                str(args.training_sample_ratio) + '.mat')
    
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' + 
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda:0')
    for fold_idx in range(folds_num):
        fold_idx=fold_idx
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = False,num_workers=4)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=4)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        global all_z,all_label,all_inc_V,all_inc_L
        all_z = torch.tensor([]).to('cuda:0')
        all_label,all_inc_V,all_inc_L = torch.tensor([]).to('cuda:0'),torch.tensor([]).to('cuda:0'),torch.tensor([]).to('cuda:0')
        labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
        dep_graph = torch.matmul(labels.T,labels)
        dep_graph = dep_graph/(torch.diag(dep_graph).unsqueeze(1)+1e-10)
        dep_graph[dep_graph<=args.sigma]=0.
        dep_graph.fill_diagonal_(fill_value=1.)
        # dep_graph = F.softmax(dep_graph,dim=-1)
        model = get_model(n_stacks=4,n_input=d_list,n_z=args.n_z,Nlabel=classes_num,device= device)
        # print(model)
        loss_model = Loss(args.alpha, classes_num, device)
        # crit = nn.BCELoss()
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)

        scheduler = None
        
        
        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch=0
        best_model_dict = {'model':model.state_dict(),'epoch':0}
        for epoch in range(args.epochs):
            
            train_losses,model = train(train_dataloder,model,loss_model,optimizer,scheduler,epoch,dep_graph,logger)
            # test_results = test(test_dataloder,model,loss_model,epoch,dep_graph,logger)
            val_results = test(val_dataloder,model,loss_model,epoch,dep_graph,logger)
            # for i,re in enumerate(epoch_results):
                # re.update(test_results[i])
            
            if val_results[0]*0.25+val_results[2]*0.25+val_results[3]*0.5>=static_res:
                static_res = val_results[0]*0.25+val_results[2]*0.25+val_results[3]*0.5
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch=epoch
            train_losses_last = train_losses
            total_losses.update(train_losses.sum)
        model.load_state_dict(best_model_dict['model'])
        print("epoch",best_model_dict['epoch'])
        test_results = test(test_dataloder,model,loss_model,epoch,dep_graph,logger)

        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,best_epoch,test_results[0],test_results[1],
            test_results[2],test_results[3]))

        for i in range(9):
            folds_results[i].update(test_results[i])
        if args.save_curve:
            np.save(osp.join(args.curve_dir,args.dataset+'_V_'+str(args.mask_view_ratio)+'_L_'+str(args.mask_label_ratio))+'_'+str(fold_idx)+'.npy', np.array(list(zip(epoch_results[0].vals,train_losses.vals))))
    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP HL RL AUCme one_error coverage macAUC macro_f1 micro_f1 lr alpha beta gamma sigma\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg,4))+'+'+str(round(res.std,4)) for res in folds_results]
    res_list.extend([str(args.lr),str(args.alpha),str(args.beta),str(args.gamma),str(args.sigma)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()
        

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'newrecords'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH', 
                        default='/disk1/lcl/MATLAB-NOUPLOAD/MyMVML-data/')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--name', type=str, default='10_newfinal_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=250)
    
    # Training args
    parser.add_argument('--n_z', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--sigma', type=float, default=0.)

    
    args = parser.parse_args()

    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e-1]
    alpha_list = [1e0]#[1e0,1e1,1e2,1e-1,1e-2,1e-3] #1e2 for pascal07  1e1 for others
    beta_list = [0.97]  
    gamma_list = [1e-1] #for all dataset with double 0.5 missing rates
    sigma_list = [1] #0 for corel5k and pascal07; 0.5 for espgame and iaprtc12; 0.5/1 for mirflickr (for both 0.5 missing rates  missing rates)
    if args.lr >= 0.01:
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for sigma in sigma_list:
                        args.sigma = sigma
                        for dataset in args.datasets:
                            args.dataset = dataset
                            file_path = osp.join(args.records_dir,args.name+args.dataset+'_ViewMask_' + str(
                                            args.mask_view_ratio) + '_LabelMask_' +
                                            str(args.mask_label_ratio) + '_Training_' + 
                                            str(args.training_sample_ratio) + '.txt')
                            args.file_path = file_path
                            existed_params = filterparam(file_path,[-4,-3,-2,-1])
                            if [args.alpha,args.beta,args.gamma] in existed_params:
                                print('existed param! alpha:{} beta:{} gamma:{} sigma:{}'.format(args.alpha,args.beta,args.gamma,args.sigma))
                                # continue
                            main(args,file_path)