import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, t, Nlabel, device):
        super(Loss, self).__init__()

        self.Nlabel = Nlabel
        self.t = t
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mse = nn.MSELoss()
    def grather_positive_pairs(self,emb_list,inc_V_ind):
        loss = 0
        d = emb_list[0].shape[-1]
        emb_nor_list = []
        for emb in emb_list:
            emb_nor_list.append(F.normalize(emb, p=2, dim=-1))
        for i in range(len(emb_nor_list)):
            for j in range(i+1,len(emb_nor_list)):
                vaild_ind_ij = inc_V_ind[:,i].mul(inc_V_ind[:,j]).unsqueeze(-1)
                loss_ij = (emb_nor_list[i]-emb_nor_list[j]).square().mul(vaild_ind_ij).sum()
                loss += loss_ij/(d*vaild_ind_ij.sum()+1e-10)
        return loss
        
    

    def wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret

    
    def all_label_guide(self, x, inc_labels, inc_V_ind, inc_L_ind, all_x, all_labels, all_V_ind, all_L_ind, i, bs):
        x = torch.stack(x,dim=1) if isinstance(x,list) else x
        all_x = torch.stack(all_x,dim=1) if isinstance(all_x,list) else all_x
        n = x.size(0)
        N = all_x.size(0)
        v = x.size(1)
        if n == 1:
            return 0
        valid_labels_sum = torch.matmul(inc_L_ind.float(), all_L_ind.float().T) #[n, N] 
        # all_x[i*bs:i*bs+n:,:,:] = x.detach().clone()
        labels = (torch.matmul(inc_labels, all_labels.T) / (valid_labels_sum + 1e-9)) #[n,N]
        labels[:,i*bs:i*bs+n] = labels[:,i*bs:i*bs+n].fill_diagonal_(0) #[n,N]
        # labels = torch.softmax(labels.masked_fill(labels==0,-1e9),dim=-1)
        labels = labels/(labels.max(dim=-1).values.unsqueeze(-1)+1e-9)
        x = F.normalize(x, p=2, dim=-1)
        x = x.transpose(0,1) #[v,n,d]
        all_x = F.normalize(all_x, p=2, dim=-1)
        all_x = all_x.transpose(0,1) #[v,N,d]
        all_x_T = torch.transpose(all_x,-1,-2)#[v,d,N]
        # sim = torch.abs(torch.matmul(x,all_x_T)) # [v, n, N]
        sim = (1+torch.matmul(x,all_x_T))/2 # [v, n, N]
        sim[sim>1]=1
        mask_v = (inc_V_ind.T).unsqueeze(-1).mul((all_V_ind.T).unsqueeze(1)) #[v, n, N]
        mask_v[:,:,i*bs:i*bs+n] = mask_v[:,:,i*bs:i*bs+n].masked_fill(torch.eye(n,device=x.device)==1,0.)
        assert torch.sum(torch.isnan(mask_v)).item() == 0
        assert torch.sum(torch.isnan(labels)).item() == 0
        assert torch.sum(torch.isnan(sim)).item() == 0
        # print('labels',torch.sum(torch.max(labels)))
        # loss = ((sim.view(v,-1)-labels.view(1,n*n))**2).mul(mask_v.view(v,-1)) # sim labels view [v, n* n]

        loss = self.weighted_BCE_loss(sim.view(v,-1),labels.view(1,n*N).expand(v,-1),mask_v.view(v,-1),reduction='none')
        # assert torch.sum(torch.isnan(loss)).item() == 0
        
        loss = loss.sum(dim=-1)/(mask_v.view(v,-1).sum(dim=-1))
        return 0.5*loss.sum()/v
    
    def New_CE9(self,pre,label,inc_L_ind,dep_graph,reduction='mean'):
        pre = torch.clamp(pre,1e-10,1.-1e-1)
        # print(pre.min(),pre.max())
        # print('pre',torch.sum(torch.isnan(torch.log(pre))).item())
        # print('1-pre',torch.sum(torch.isnan(torch.log(1 - pre))).item())

        loss = torch.mean(torch.abs((label.mul(torch.log(pre).mul(inc_L_ind).mm(dep_graph.T)) \
                                + (1-label).mul(torch.log(1 - pre ).mul(inc_L_ind).mm(dep_graph))).mul(inc_L_ind)))
        # print('1-pre',torch.sum(torch.isnan(torch.log(1 - pre))).item())
        return loss
    
    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0

        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res



    
