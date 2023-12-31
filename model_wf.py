import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dims,use_bias = True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_dim,out_dims))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_dims))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
    def forward(self, A, X):
        support = torch.mm(X, self.weight)
        out = torch.mm(A,support)
        if self.use_bias:
            out = out + self.bias
        return out

class GCNNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim=1024,output_dim=1024,num_layers=2):
        super(GCNNet, self).__init__()
        dims = [(input_dim,input_dim) for n in range(num_layers-1)]
        dims.append((input_dim,output_dim))
        self.gcnList = nn.ModuleList([GCNLayer(input_dim,output_dim) for (input_dim,output_dim) in dims])
        self.gcn1 = GCNLayer(input_dim, input_dim)
        self.gcn2 = GCNLayer(input_dim, output_dim)
    
    def forward(self, adjacency, x):
        # for i in range(len(self.gcnList)-1):
        #     x = F.relu(self.gcnList[i](adjacency, x))
        # logits = self.gcnList[-1](adjacency, x)
        x = self.gcn1(adjacency, x)
        logits = self.gcn2(adjacency, x)
        return logits


# class encoder(nn.Module):
#     def __init__(self, n_dim, dims, n_z):
#         super(encoder, self).__init__()
#         # print(n_dim,dims[0])
#         self.enc_0 = Linear(n_dim, dims[0])
#         self.enc_1 = Linear(dims[0], dims[1])
#         self.enc_2 = Linear(dims[1], dims[2])
#         self.enc_3 = Linear(dims[2], n_z)
#         self.enc_4 = Linear(n_z, n_z)
#         self.z_b0 = nn.BatchNorm1d(n_z)

#     def forward(self, x):
#         enc_h0 = F.relu(self.enc_0(x))
#         enc_h1 = F.relu(self.enc_1(enc_h0))
#         enc_h2 = F.relu(self.enc_2(enc_h1))
#         enc_h3 = F.relu(self.enc_3(enc_h2))
#         z = self.z_b0(self.enc_4(enc_h3))
#         return z
class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        self.enc_3 = Linear(dims[1], dims[2])
        self.z_layer = Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h3))
        return z

class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        # self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        # r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar


class AE(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(AE, self).__init__()


        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.view_num = len(n_input)
        self.regression = Linear(n_z, nLabel)
        self.act = nn.Sigmoid()
        self.nLabel = nLabel
        self.labelword = nn.Parameter(torch.randn(nLabel, n_z)).cuda()
        # self.classifier_list = nn.ModuleList([Linear(n_z, nLabel) for v in range(len(n_input))])
        self.labelgcn = GCNNet(128,128,3)
        self.discriminator = nn.Sequential(nn.Linear(n_z,n_z),
            nn.ReLU(),
            nn.Linear(n_z, 1))
        # self.regression2 = Linear(nLabel, nLabel)
    def forward(self, mul_X, we):
        # dep_graph = torch.eye(self.nLabel,device=we.device).float()
        batch_size = mul_X[0].shape[0]
        summ = 0
        individual_zs = []
        for enc_i, enc in enumerate(self.encoder_list):
            z_i = enc(mul_X[enc_i])
            individual_zs.append(z_i)
            # summ += torch.diag(we[:, enc_i]).mm(z_i)
        zs_emb = torch.stack(individual_zs,dim=1) #[n v d]
        # dis_score = self.discriminator(zs_emb.detach()).squeeze(-1) #[n v]
        dis_score = self.discriminator(zs_emb).squeeze(-1) #[n v]
        # dis_score[(1-we).bool()] = -1e9
        dis_score = F.softmax(dis_score,dim=-1)
        # z = zs_emb.mul(we.unsqueeze(-1)).sum(1)/(we.sum(1,keepdim=True))
        z = zs_emb.mul(dis_score.unsqueeze(-1).detach()).sum(1)
        # wei = 1 / torch.sum(we, 1)
        # z = torch.diag(wei).mm(summ)
        
        x_bar_list = []
        for dec_i, dec in enumerate(self.decoder_list):
            x_bar_list.append(dec(individual_zs[dec_i]))
            # x_bar_list.append(dec(z))


        individual_preds = self.act(self.regression(F.relu(zs_emb))).detach() #[n v c]
        # print(individual_preds.min())

        logi = self.regression(F.relu(z)) #[n c]
        # logi = self.labelgcn(dep_graph,logi.T).T
        # logi = F.relu(z).mm(W.T)
        yLable = self.act(logi)
        return x_bar_list, yLable, z, individual_zs, individual_preds, dis_score


class net(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 Nlabel):
        super(net, self).__init__()

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z,
            nLabel=Nlabel)

    def forward(self, mul_X, we):
        

        return self.ae(mul_X, we)

def get_model(n_stacks,n_input,n_z,Nlabel,device):
    model = net(n_stacks=n_stacks,n_input=n_input,n_z=n_z,Nlabel=Nlabel).to(device)
    return model