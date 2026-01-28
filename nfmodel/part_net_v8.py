# @Time    : 06/05/2021
# @Author  : Wei Chen
# @Project : Pycharm
import torch.nn as nn
import numpy as np
np.bool=np.bool_
np.float=np.float_
np.int=np.int_
import nfmodel.gcn3d as gcn3d
import nfmodel.equ_gcn3d as equi_gcn
from nfmodel.equ_gcn3d import *
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as functional
from eqnet.models.query_producer.qnet import *
import absl.flags as flags
FLAGS = flags.FLAGS
from nfmodel.neuron_net import MyVecDGCNN_att
from occ_net import ConvPointnet
from nfmodel.vec_layers import *
from model_cl import MyCLNet,score_model






class Pair_loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.lin3=VNLinear(256,256)
        self.bn3=VNLayerNorm(256)
        self.relu3=VNReLU(256)
        self.lin4=VNLinear(256,1)
        self.loss_func= nn.SmoothL1Loss(beta=0.5)
    def forward(self,z_so3,vertices):
        z_so3_2=z_so3[:,1:]
        vertices_2=vertices[:,1:]

        z_so3=z_so3[:,:1].repeat(1,z_so3_2.shape[1],1,1)
        vertices=vertices[:,:1].repeat(1,z_so3_2.shape[1],1)
        pair_z_so3=torch.cat([z_so3,z_so3_2],dim=2)
        pair_dir=vertices_2-vertices
        x = self.relu3(self.bn3(self.lin3(pair_z_so3)))
        x=self.lin4(x).squeeze(2)
        loss=self.loss_func(x,pair_dir)
        return loss
    def generate(self,z_so3,z_so3_2):
        pair_z_so3=torch.cat([z_so3,z_so3_2],dim=2)
        x = self.relu3(self.bn3(self.lin3(pair_z_so3)))
        x=self.lin4(x).squeeze(2)
        return x

class Pair_loss_nocs(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 3, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.loss_func= nn.SmoothL1Loss(beta=0.5)
    def forward(self,z_inv,nocs):
        z_inv_2=z_inv[:,1:]
        nocs_2=nocs[:,1:]

        z_inv=z_inv[:,:1].repeat(1,z_inv_2.shape[1],1)
        nocs=nocs[:,:1].repeat(1,z_inv_2.shape[1],1)

        pair_z_inv=torch.cat([z_inv,z_inv_2],dim=2)
        pair_dir=nocs_2-nocs
        x = F.relu(self.bn3(self.conv3(pair_z_inv.permute(0,2,1))))
        x=self.conv4(x).permute(0,2,1)
        loss=self.loss_func(x,pair_dir)
        return loss
    def generate(self,z_inv,z_inv_2):
        pair_z_inv=torch.cat([z_inv,z_inv_2],dim=2)
        x = F.relu(self.bn3(self.conv3(pair_z_inv.permute(0,2,1))))
        x=self.conv4(x).permute(0,2,1)
        return x


class Rot_per_equi(nn.Module):
    def __init__(self):
        super(Rot_per_equi, self).__init__()

        self.lin1=VNLinear(128,256)
        self.bn1=VNLayerNorm(256)
        self.relu1=VNReLU(256)
        self.lin2=VNLinear(256,256)
        self.bn2=VNLayerNorm(256)
        self.bn2=VNLayerNorm(256)
        self.relu2=VNReLU(256)
        self.lin4=VNLinear(256,1)
        self.loss_func= nn.SmoothL1Loss(beta=0.5)

    def forward(self, x,gt_v):
        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.bn2(self.lin2(x)))
        x=self.lin4(x).squeeze(2)
        loss=self.loss_func(x,gt_v.unsqueeze(1))
        return loss

    def generate(self,x):
        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.bn2(self.lin2(x)))
        x=self.lin4(x).squeeze(2)
        return x

class Weight_model2(nn.Module):
    def __init__(self):
        super(Weight_model2, self).__init__()
        self.weight_model=score_model()
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., dtype=torch.float32),
            torch.tensor(1.,  dtype=torch.float32))
        self.num_samples=50
        self.tau = 1
        self.ratio=0.1
        self.max_iters=5
        self.batch_size=20
        # self.ratio_loss_fun=nn.SmoothL1Loss(beta=0.5,reduction='mean')
    def forward(self,fea,gt_camera_batched,inv_sigma_batched):
        query_num=fea.shape[1]
        # fea=fea.permute(0,2,1)[:,:,:,None]
        logits_batched=self.weight_model(fea)
        bs=fea.shape[0]
        batch_min_values_mean=[]
        batch_min_stds_mean=[]
        batch_min_ratios_mean=[]
        for i in range(bs):
            logits=logits_batched[i:i+1].repeat(self.batch_size,1,1)
            gt_camera=gt_camera_batched[i:i+1]
            inv_sigma=inv_sigma_batched[i:i+1]
            min_values=[]
            min_stds=[]
            min_ratios=[]
            for iters in range(self.max_iters):
                pred_mask=F.gumbel_softmax(logits,hard=True,dim=-1)[:,:,0]
                ret=pred_mask


                gt_camera_min=ret[:,:,None]*gt_camera
                pred_mask_sum=torch.sum(ret,dim=-1,keepdim=True)
                pred_mask_ratio=pred_mask_sum/query_num
                inv_sigma_min=ret[:,:,None,None]*inv_sigma

                p=gt_camera_min
                p_center=torch.sum(p,dim=1,keepdim=True)/(pred_mask_sum.detach())[:,:,None]
                p=p-p_center
                p_norm=torch.norm(p,dim=-1,keepdim=True)*ret.detach()[:,:,None]
                p_norm_mean=torch.sum(p_norm,dim=1,keepdim=True)/(pred_mask_sum.detach())[:,:,None]
                p=p/(p_norm_mean+1e-6)*ret.detach()[:,:,None]
                FT=torch.zeros(self.batch_size,query_num,3,6)
                h=hat(p)
                c=torch.einsum('psij,psjk->psik',-inv_sigma_min,h)
                FT[:,:,:,:3]=c
                FT[:,:,:,3:]=inv_sigma_min
                C=torch.einsum('psij,psik->pjk',FT,FT)
                try:
                    values=torch.linalg.eigvalsh(C)
                    values=(values[:,0]/((values[:,-1]).detach())).mean()
                except:
                    print('invalid eigen')
                    values=0

                ratio_min=(pred_mask_ratio-self.ratio)**2
                min_values.append(values)
                min_ratios.append(ratio_min.mean())

            min_values_mean=sum(min_values)/self.max_iters
            min_stds_mean=sum(min_stds)/self.max_iters
            min_ratios_mean=sum(min_ratios)/self.max_iters
            batch_min_values_mean.append(min_values_mean)
            batch_min_stds_mean.append(min_stds_mean)
            batch_min_ratios_mean.append(min_ratios_mean)


        return sum(batch_min_values_mean)/bs,sum(batch_min_ratios_mean)/bs

    def generate(self,fea):
        logits_batched=self.weight_model(fea)
        pred_mask=F.gumbel_softmax(logits_batched,hard=True,dim=-1)[:,:,0]
        pred_mask_sum=torch.sum(pred_mask,dim=-1)
        return pred_mask


class Weight_model(nn.Module):
    def __init__(self):
        super(Weight_model, self).__init__()
        self.weight_model=MyCLNet()
        self.num_samples =3
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., dtype=torch.float32),
            torch.tensor(1.,  dtype=torch.float32))
        self.tau = 1
        self.max_iters=50
        self.batch_size=32
        self.loss_func= nn.SmoothL1Loss(beta=0.5)
        self.pool=VNMaxPool()

    def forward(self, z_inv_batched,z_so3_batched,rot_net1,rot_net2):
        bs=z_inv_batched.shape[0]
        rot_net1=rot_net1.eval()
        rot_net2=rot_net2.eval()
        z_inv_batched=z_inv_batched.permute(0,2,1).unsqueeze(-1)
        logits_batched=self.weight_model(z_inv_batched)
        z_so3_1_batched=rot_net1.before_pool(z_so3_batched).detach()
        z_so3_2_batched=rot_net2.before_pool(z_so3_batched).detach()

        z_so3_1_mean_batched=self.pool(z_so3_1_batched.permute(0,2,3,1)).detach()
        z_so3_2_mean_batched=self.pool(z_so3_2_batched.permute(0,2,3,1)).detach()

        avg_loss_batched=0
        for i in range(bs):
            logits=logits_batched[i:i+1].repeat(self.batch_size,1)
            z_so3_1=z_so3_1_batched[i:i+1]
            z_so3_2=z_so3_2_batched[i:i+1]
            z_so3_1_mean=z_so3_1_mean_batched[i:i+1]
            z_so3_2_mean=z_so3_2_mean_batched[i:i+1]
            avg_loss=0
            for  iters in range(self.max_iters):
                gumbels = self.gumbel_dist.sample(logits.shape).to(logits.device)
                gumbels = (logits + gumbels)/self.tau
                y_soft = gumbels.softmax(-1)
                topk = torch.topk(gumbels, self.num_samples, dim=-1)
                y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, topk.indices, 1.0)
                ret = y_hard - y_soft.detach() + y_soft
                z_so3_1_minimal=z_so3_1*ret[:,:,None,None]
                z_so3_2_minimal=z_so3_2*ret[:,:,None,None]
                z_so3_1_minimal=z_so3_1_minimal[ret!=0]\
                    .reshape(self.batch_size,self.num_samples,-1,3)
                z_so3_2_minimal=z_so3_2_minimal[ret!=0] \
                    .reshape(self.batch_size,self.num_samples,-1,3)
                z_so3_1_minimal_mean=self.pool(z_so3_1_minimal.permute(0,2,3,1))
                z_so3_2_minimal_mean=self.pool(z_so3_2_minimal.permute(0,2,3,1))
                square_distance_1=self.loss_func(z_so3_1_minimal_mean,z_so3_1_mean)
                square_distance_2=self.loss_func(z_so3_2_minimal_mean,z_so3_2_mean)
                avg_loss+=square_distance_1+square_distance_2
            avg_loss/=self.max_iters
            avg_loss_batched+=avg_loss
        return avg_loss_batched/bs




class Pose_Ts(nn.Module):
    def __init__(self,k=6,F=1283):
        super(Pose_Ts, self).__init__()
        self.f=F
        self.k = k

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs





class GCN3D_segR(nn.Module):
    def __init__(self, support_num, neighbor_num):
        super(GCN3D_segR, self).__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 128, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num= support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.vecnum = 3
        dim_fuse = sum([128, 128, 256, 256, 512])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, self.vecnum, 1),
        )

    def forward(self,vertices: "tensor (bs, vetice_num, 3)"):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace= True)


        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1,2)).transpose(1,2), inplace= True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1,2)).transpose(1,2), inplace= True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1,2)).transpose(1,2), inplace= True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                     v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0] #(bs, f)
        feature_dict={'conv0':{'pos':vertices,'fea':fm_0},
                      'conv1':{'pos':vertices,'fea':fm_1},
                      'conv2':{'pos':v_pool_1,'fea':fm_2},
                      'conv3':{'pos':v_pool_1,'fea':fm_3},
                      'conv4':{'pos':v_pool_2,'fea':fm_4},

                      }

        return feature_dict


class Point_center(nn.Module):
    def __init__(self):
        super(Point_center, self).__init__()

        # self.conv1 = torch.nn.Conv2d(12, 64, 1) ##c
        self.conv1 = torch.nn.Conv1d(3, 128, 1) ## no c
        self.conv2 = torch.nn.Conv1d(128, 256, 1)

        ##here
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        # self.conv4 = torch.nn.Conv1d(1024,1024,1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        # self.bn4 = nn.BatchNorm1d(1024)
        # self.global_feat = global_feat

    def forward(self, x):## 5 6 30 1000


        x = F.relu(self.bn1(self.conv1(x))) ## 5 64 30 1000
        x = F.relu(self.bn2(self.conv2(x))) ## 5 64 1 1000
        x = (self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        x2 = torch.max(x, -1, keepdim=True)[0]#5 512 1
        # x2=torch.mean(x, -1, keepdim=True)


        return x2


class Point_center_res_cate(nn.Module):
    def __init__(self):
        super(Point_center_res_cate, self).__init__()

        # self.feat = Point_vec_edge()
        self.feat = Point_center()
        self.conv1 = torch.nn.Conv1d(512, 256,1)
        self.conv2 = torch.nn.Conv1d(256, 128,1)
        # self.drop1 = nn.Dropout(0.1)
        self.conv3 = torch.nn.Conv1d(128, 6,1 )


        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = (self.bn2(self.conv2(x)))

        x=self.drop1(x)
        x = self.conv3(x)



        x = x.squeeze(2)
        x=x.contiguous()##Bx6
        xt = x[:,0:3]
        xs = x[:,3:6]
        if torch.isnan(xs).any():
            print('asd')
        return xt,xs

class Rot_green(nn.Module):
    def __init__(self, k=3,F=1280):
        super(Rot_green, self).__init__()
        self.f=F
        self.k = k


        self.conv1 = torch.nn.Conv1d(self.f , 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256,256,1)
        self.conv4 = torch.nn.Conv1d(256,self.k,1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x=self.drop1(x)
        x = self.conv4(x)

        x=x.squeeze(2)
        x = x.contiguous()


        return x

class Rot_red(nn.Module):
    def __init__(self, k=24,F=1036):
        super(Rot_red, self).__init__()
        self.f=F
        self.k = k

        self.conv1 = torch.nn.Conv1d(self.f , 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256,256,1)
        self.conv4 = torch.nn.Conv1d(256,self.k,1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x=self.drop1(x)
        x = self.conv4(x)

        x=x.squeeze(2)
        x = x.contiguous()


        return x






class T_equi(nn.Module):
    def __init__(self,dim=128+1):
        super(T_equi, self).__init__()

        self.lin1=VNLinear(dim,256)
        self.bn1=VNLayerNorm(256)
        self.relu1=VNReLU(256)
        self.lin2=VNLinear(256,256)
        self.bn2=VNLayerNorm(256)
        self.relu2=VNReLU(256)
        self.pool=VNMaxPool()

        self.lin3=VNLinear(256,256)
        self.bn3=VNLayerNorm(256)
        self.relu3=VNReLU(256)
        self.lin4=VNLinear(256,1)


    def forward(self, x,mask=None):
        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.bn2(self.lin2(x)))
        x=x.permute(0,2,3,1)
        # x=torch.mean(x,dim=-1)
        x=self.pool(x,mask)
        x = self.relu3(self.bn3(self.lin3(x)))
        x=self.lin4(x)
        t=x[:,0,:]
        return t

    def before_pool(self,xmask=None):
        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.bn2(self.lin2(x)))
        return x


class S_equi(nn.Module):
    def __init__(self,dim=128+1):
        super(S_equi, self).__init__()

        self.lin1=VNLinear(dim,256)
        self.bn1=VNLayerNorm(256)
        self.relu1=VNReLU(256)
        self.lin2=VNLinear(256,256)
        self.bn2=VNLayerNorm(256)
        self.relu2=VNReLU(256)
        self.pool=VNMaxPool()

        self.lin3=VNLinear(256,256)
        self.bn3=VNLayerNorm(256)
        self.relu3=VNReLU(256)
        self.lin4=VNLinear(256,3)
        self.lin5=VNLinear(256,3)


    def forward(self, x,mask=None):
        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.bn2(self.lin2(x)))
        x=x.permute(0,2,3,1)
        # x=torch.mean(x,dim=-1)
        x=self.pool(x,mask)
        x = self.relu3(self.bn3(self.lin3(x)))
        x1=self.lin4(x)
        x2=self.lin5(x)
        s=(x1*x2).sum(-1)
        return s





class Rot_green_equi(nn.Module):
    def __init__(self,dim=128):
        super(Rot_green_equi, self).__init__()

        self.lin1=VNLinear(dim,dim)
        self.bn1=VNLayerNorm(dim)
        self.relu1=VNReLU(dim)
        self.lin2=VNLinear(dim,1024)
        self.bn2=VNLayerNorm(1024)
        self.relu2=VNReLU(1024)
        self.pool=VNMaxPool()

        self.lin3=VNLinear(1024,512)
        self.bn3=VNLayerNorm(512)
        self.relu3=VNReLU(512)
        self.lin4=VNLinear(512,256)
        self.bn4=VNLayerNorm(256)
        self.relu4=VNReLU(256)
        self.lin5=VNLinear(256,1)


    def forward(self, x,mask=None):
        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.bn2(self.lin2(x)))
        x=x.permute(0,2,3,1)
        # x=torch.mean(x,dim=-1)
        x=self.pool(x,mask)
        x = self.relu3(self.bn3(self.lin3(x)))
        x = self.relu4(self.bn4(self.lin4(x)))
        x=self.lin5(x).squeeze(1)

        return x


class Rot_red_equi(nn.Module):
    def __init__(self, dim=128):
        super(Rot_red_equi, self).__init__()

        self.lin1=VNLinear(dim,dim)
        self.bn1=VNLayerNorm(dim)
        self.relu1=VNReLU(dim)
        self.lin2=VNLinear(dim,1024)
        self.bn2=VNLayerNorm(1024)
        self.relu2=VNReLU(1024)
        self.pool=VNMaxPool()

        self.lin3=VNLinear(1024,512)
        self.bn3=VNLayerNorm(512)
        self.relu3=VNReLU(512)
        self.lin4=VNLinear(512,256)
        self.bn4=VNLayerNorm(256)
        self.relu4=VNReLU(256)
        self.lin5=VNLinear(256,1)


    def forward(self, x,mask=None):
        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.bn2(self.lin2(x)))
        x=x.permute(0,2,3,1)
        # x=torch.mean(x,dim=-1)
        x=self.pool(x,mask)
        x = self.relu3(self.bn3(self.lin3(x)))
        x = self.relu4(self.bn4(self.lin4(x)))
        x=self.lin5(x).squeeze(1)

        return x





def square_distance(src, dst):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py

    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim_global,dim_inp, dim, nneigh=7, reduce_dim=True, separate_delta=True):
        super().__init__()

        # dim_inp = dim
        # dim = dim  # // 2
        self.dim = dim

        self.nneigh = nneigh
        self.separate_delta = separate_delta

        self.fc_delta = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        #if self.separate_delta:
        #    self.fc_delta2 = nn.Sequential(
        #        nn.Linear(3, dim),
        #        nn.ReLU(),
        #        nn.Linear(dim, dim)
        #
        #    )

        self.fc_gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.w_k_global = nn.Linear(dim_global, dim, bias=False)
        self.w_v_global = nn.Linear(dim_global, dim, bias=False)

        self.w_qs = nn.Linear(dim_global, dim, bias=False)

        self.w_ks = nn.Linear(dim_inp, dim, bias=False)
        self.w_vs = nn.Linear(dim_inp, dim, bias=False)

        if not reduce_dim:
            self.fc = nn.Linear(dim, dim_inp)
        self.reduce_dim = reduce_dim

    # xyz_q: B x n_queries x 3
    # lat_rep: B x dim
    # xyz: B x n_anchors x 3,
    # points: B x n_anchors x dim
    def forward(self, xyz_q, lat_rep, xyz, points):
        with torch.no_grad():
            dists = square_distance(xyz_q, xyz)
            ## knn group
            knn_idx = dists.argsort()[:, :, :self.nneigh]  # b x nQ x k
            #print(knn_idx.shape)

            #knn = KNN(k=self.nneigh, transpose_mode=True)
            #_, knn_idx = knn(xyz, xyz_q)  # B x npoint x K
            ##
            #print(knn_idx.shape)

        b, nQ, _ = xyz_q.shape
        # b, nK, dim = points.shape

        if len(lat_rep.shape) == 2:
            q_attn = self.w_qs(lat_rep).unsqueeze(1).repeat(1, nQ, 1)
            k_global = self.w_k_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
        else:
            q_attn = self.w_qs(lat_rep)
            k_global = self.w_k_global(lat_rep).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(2)

        k_attn = index_points(self.w_ks(points),
                              knn_idx)  # b, nQ, k, dim  # self.w_ks(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        k_attn = torch.cat([k_attn, k_global], dim=2)
        v_attn = index_points(self.w_vs(points), knn_idx)  # #self.w_vs(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        v_attn = torch.cat([v_attn, v_global], dim=2)
        xyz = index_points(xyz, knn_idx)  # xyz = xyz.unsqueeze(1).repeat(1, nQ, 1, 1)
        pos_encode = self.fc_delta(xyz_q[:, :, None] - xyz)  # b x nQ x k x dim
        pos_encode = torch.cat([pos_encode, torch.zeros([b, nQ, 1, self.dim], device=pos_encode.device)],
                               dim=2)  # b, nQ, k+1, dim
        if self.separate_delta:
            pos_encode2 = self.fc_delta(xyz_q[:, :, None] - xyz)  # b x nQ x k x dim
            pos_encode2 = torch.cat([pos_encode2, torch.zeros([b, nQ, 1, self.dim], device=pos_encode2.device)],
                                    dim=2)  # b, nQ, k+1, dim
        else:
            pos_encode2 = pos_encode

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x nQ x k+1 x dim

        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode2)  # b x nQ x dim

        if not self.reduce_dim:
            res = self.fc(res)
        return res

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Copied from https://github.com/autonomousvision/convolutional_occupancy_networks

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class PointTransformerDecoderOcc(nn.Module):
    """
    AIR-Net decoder

    Attributes:
        dim_inp int: dimensionality of encoding (global and local latent vectors)
        dim int: internal dimensionality
        nneigh int: number of nearest anchor points to draw information from
        hidden_dim int: hidden dimensionality of final feed-forward network
        n_blocks int: number of blocks in feed forward network
    """
    def __init__(self, dim_global,dim_inp=1280, dim=200, nneigh=7, hidden_dim=64, n_blocks=5):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks


        self.ct1 = CrossTransformerBlock(dim_global,dim_inp, dim, nneigh=nneigh)
        #self.fc_glob = nn.Linear(dim_inp, dim)

        self.init_enc = nn.Linear(dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, 7)

        self.actvn = F.relu
        self.scale_branch = nn.Linear(512, 3)
    def forward(self, xyz_q, encoding):
        """
        TODO update commont to include encoding dict
        :param xyz_q [B x n_queries x 3]: queried 3D coordinates
        :param lat_rep [B x dim_inp]: global latent vectors
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchros x dim_inp]: local latent vectors
        :return: occ [B x n_queries]: occupancy probability for each queried 3D coordinate
        """

        lat_rep = encoding['z']
        global_feat=lat_rep
        xyz = encoding['anchors']
        feats = encoding['anchor_feats']

        lat_rep = self.ct1(xyz_q, lat_rep, xyz, feats)  # + self.fc_glob(lat_rep).unsqueeze(1).repeat(1, xyz_q.shape[1], 1) +
        net = self.init_enc(lat_rep)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)
        out= self.fc_out(self.actvn(net))
        nocs =out[:,:,:3]
        occ=nn.functional.sigmoid(out[:,:,3:4])
        w3d=out[:,:,4:7]
        scale = self.scale_branch(global_feat).exp()
        return nocs,occ,lat_rep,w3d,scale


class DampingNet(nn.Module):
    def __init__(self, num_params=9,min=-6,max=5):
        super().__init__()
        const = torch.zeros(num_params)
        self.min_=min
        self.max_=max
        self.register_parameter('const', torch.nn.Parameter(const))


    def forward(self):
        lambda_ = 10.**(self.min_ + self.const.sigmoid()*(self.max_ - self.min_))
        return lambda_



class MyQNet(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_out = nn.Linear(self.q_target_chn, 3+3+3+3)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        query_features=self.qnet(data_dict=data_dict,pred_scale=pred_scale)['query_features']
        out=self.fc_out(query_features)
        pred_dict={
            'coord':out[:,:,:3],
            'log_stds':out[:,:,3:6],
            'rot_vec_1':out[:,:,6:9],
            'rot_vec_2':out[:,:,9:12],
        }
        return pred_dict

class MyQNet_v6(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet_v6(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_out = nn.Linear(self.q_target_chn, 3+3+3+3)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        query_features=self.qnet(data_dict=data_dict,pred_scale=pred_scale)['query_features']
        out=self.fc_out(query_features)
        pred_dict={
            'coord':out[:,:,:3],
            'log_stds':out[:,:,3:6],
            'rot_vec_1':out[:,:,6:9],
            'rot_vec_2':out[:,:,9:12],
            'z_inv':query_features
        }
        return pred_dict










class MyQNet_equi(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet_equi(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_inv = VNLinear(self.q_target_chn, self.q_target_chn)
        self.fc_out_so3 = VNLinear(self.q_target_chn, 3)
        self.fc_out_inv = nn.Linear(self.q_target_chn, 3)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-2]*fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        query_features=self.qnet(data_dict=data_dict,pred_scale=pred_scale)['query_features']
        dual_query_features=self.fc_inv(query_features)
        inv_fea= (query_features * dual_query_features).sum(-1)
        out_inv=self.fc_out_inv(inv_fea)
        out_so3=self.fc_out_so3(query_features)
        pred_dict={
            'coord':out_so3[:,:,0,:],
            'rot_vec_1':out_so3[:,:,1,:],
            'rot_vec_2':out_so3[:,:,2,:],
            'log_stds':out_inv,

            'z_so3':query_features,
            'z_inv':inv_fea
        }
        return pred_dict


class MyQNet_equi_v3(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet_equi_v3(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_inv = VNLinear(self.q_target_chn, self.q_target_chn)
        self.fc_out_so3 = VNLinear(self.q_target_chn, 3)
        self.fc_out_inv = nn.Linear(self.q_target_chn, 3)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-2]*fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        query_features=self.qnet(data_dict=data_dict,pred_scale=pred_scale)['query_features']
        dual_query_features=self.fc_inv(query_features)
        inv_fea= (query_features * dual_query_features).sum(-1)
        out_inv=self.fc_out_inv(inv_fea)
        out_so3=self.fc_out_so3(query_features)
        pred_dict={
            'coord':out_so3[:,:,0,:],
            'rot_vec_1':out_so3[:,:,1,:],
            'rot_vec_2':out_so3[:,:,2,:],
            'log_stds':out_inv,

            'z_so3':query_features,
            'z_inv':inv_fea
        }
        return pred_dict


class MyQNet_equi_v4(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet_equi_v4(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_inv = VNLinear(self.q_target_chn, self.q_target_chn)
        self.fc_out_so3 = VNLinear(self.q_target_chn, 3)
        self.fc_out_inv = nn.Linear(self.q_target_chn, 3)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-2]*fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        out_put_dict=self.qnet(data_dict=data_dict,pred_scale=pred_scale)
        query_features=out_put_dict['query_features']
        relu_features=out_put_dict['relu_fea']
        dual_query_features=self.fc_inv(query_features)
        inv_fea= (query_features * dual_query_features).sum(-1)
        out_inv=self.fc_out_inv(inv_fea)
        out_so3=self.fc_out_so3(query_features)
        pred_dict={
            'coord':out_so3[:,:,0,:],
            'rot_vec_1':out_so3[:,:,1,:],
            'rot_vec_2':out_so3[:,:,2,:],
            'log_stds':out_inv,

            'z_so3':relu_features,
            'z_inv':inv_fea
        }
        return pred_dict

class MyQNet_equi_v5(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet_equi_v4(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_inv = VNLinear(self.q_target_chn, self.q_target_chn)
        # self.fc_out_so3 = VNLinear(self.q_target_chn, 3)
        self.fc_out_inv = nn.Linear(self.q_target_chn, 3*4)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-2]*fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        out_put_dict=self.qnet(data_dict=data_dict,pred_scale=pred_scale)
        query_features=out_put_dict['query_features']
        relu_features=out_put_dict['relu_fea']
        dual_query_features=self.fc_inv(query_features)
        inv_fea= (query_features * dual_query_features).sum(-1)
        out_inv=self.fc_out_inv(inv_fea)

        pred_dict={
            'coord': out_inv[:,:,0:3],
            'rot_vec_1': out_inv[:,:,3:6],
            'rot_vec_2': out_inv[:,:,6:9],
            'log_stds': out_inv[:,:,9:],

            'z_so3':relu_features,
            'z_inv':inv_fea
        }
        return pred_dict



class MyQNet_equi_v7(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet_equi_v4(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_inv = VNLinear(self.q_target_chn, self.q_target_chn)
        self.fc_out_so3 = VNLinear(self.q_target_chn, 4)
        self.fc_out_inv = nn.Linear(self.q_target_chn, 3*4)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-2]*fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        out_put_dict=self.qnet(data_dict=data_dict,pred_scale=pred_scale)
        query_features=out_put_dict['query_features']
        relu_features=out_put_dict['relu_fea']
        dual_query_features=self.fc_inv(query_features)
        inv_fea= (query_features * dual_query_features).sum(-1)
        out_inv=self.fc_out_inv(inv_fea)
        out_so3=self.fc_out_so3(query_features)

        pred_dict={
            'coord': out_inv[:,:,0:3],
            'rot_vec_1': out_inv[:,:,3:6],
            'rot_vec_2': out_inv[:,:,6:9],
            'log_stds': out_inv[:,:,9:],

            'c_coord':out_so3[:,:,0,:],
            'c_rot_vec_1':out_so3[:,:,1,:],
            'c_rot_vec_2':out_so3[:,:,2,:],
            'c_log_stds':out_so3[:,:,3,:],

            'z_so3':relu_features,
            'z_inv':inv_fea
        }
        return pred_dict





class MyQNet_equi_v8(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet_equi_v5(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_inv = VNLinear(self.q_target_chn, self.q_target_chn)
        self.fc_out_so3 = VNLinear(self.q_target_chn, 4)
        self.fc_out_inv = nn.Linear(self.q_target_chn, 3*4)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-2]*fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        out_put_dict=self.qnet(data_dict=data_dict,pred_scale=pred_scale)
        query_features=out_put_dict['query_features']
        relu_features=out_put_dict['relu_fea']
        dual_query_features=self.fc_inv(query_features)
        inv_fea= (query_features * dual_query_features).sum(-1)
        out_inv=self.fc_out_inv(inv_fea)
        out_so3=self.fc_out_so3(query_features)

        pred_dict={
            'coord': out_inv[:,:,0:3],
            'rot_vec_1': out_inv[:,:,3:6],
            'rot_vec_2': out_inv[:,:,6:9],
            'log_stds': out_inv[:,:,9:],

            'c_coord':out_so3[:,:,0,:],
            'c_rot_vec_1':out_so3[:,:,1,:],
            'c_rot_vec_2':out_so3[:,:,2,:],
            'c_log_stds':out_so3[:,:,3,:],

            'z_so3':relu_features,
            'z_inv':inv_fea
        }
        return pred_dict

class MyQNet_equi_an(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet_equi_an(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_inv = VNLinear(self.q_target_chn, self.q_target_chn)
        self.fc_out_so3 = VNLinear(self.q_target_chn, 3)
        self.fc_out_inv = nn.Linear(self.q_target_chn, 3)
    def forward(self, query_points, feature_dicts,pred_scale):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-2]*fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        out_put_dict=self.qnet(data_dict=data_dict,pred_scale=pred_scale)
        query_features=out_put_dict['query_features']
        relu_features=out_put_dict['relu_fea']
        dual_query_features=self.fc_inv(query_features)
        inv_fea= (query_features * dual_query_features).sum(-1)
        out_inv=self.fc_out_inv(inv_fea)
        out_so3=self.fc_out_so3(query_features)
        pred_dict={
            'coord':out_so3[:,:,0,:],
            'rot_vec_1':out_so3[:,:,1,:],
            'rot_vec_2':out_so3[:,:,2,:],
            'log_stds':out_inv,

            'z_so3':relu_features,
            'z_inv':inv_fea
        }
        return pred_dict





class MyNeuron(nn.Module):
    def __init__(self,):
        super(MyNeuron,self).__init__()
        self.net=MyVecDGCNN_att()


    def forward(self,vertices: "tensor (bs, vetice_num, 3)"):
        """
        Return: (bs, vertice_num, class_num)
        """
        vertices=vertices.permute(0,2,1)
        ret=self.net(vertices)
        fea_dict,pred_scale,feature_out=ret


        return fea_dict,pred_scale,feature_out.transpose(-1,-2)

class MyOccnet(nn.Module):
    def __init__(self,):
        super(MyOccnet,self).__init__()
        self.net=ConvPointnet(c_dim=256, hidden_dim=128, plane_resolution=64)
        self.fc_out = nn.Linear(256, 256)
        # self.m=nn.Sigmoid()
    def forward(self,x,query):
        shape_features = self.net(x, query)
        return self.fc_out(shape_features)


class Equi_gcn(nn.Module):
    def __init__(self, neighbor_num):
        super(Equi_gcn, self).__init__()
        self.neighbor_num = neighbor_num
        self.dim_list=[]
        for s in FLAGS.dim_list:
                self.dim_list.append(int(s))
        self.conv_0 = equi_gcn.equi_conv(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
        if FLAGS.fix_kernel:
            # self.conv_0 = equi_gcn.equi_conv2(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
            self.conv_1 = equi_gcn.equi_conv2(self.dim_list[0], self.dim_list[1])
            self.pool_1 = equi_gcn.equ_pool_layer2(pooling_rate= 4, neighbor_num= 4)
            self.conv_2 = equi_gcn.equi_conv2(self.dim_list[1], self.dim_list[2])
            self.conv_3 = equi_gcn.equi_conv2(self.dim_list[2], self.dim_list[3])
            self.pool_2 = equi_gcn.equ_pool_layer2(pooling_rate= 4, neighbor_num= 4)
            self.conv_4 = equi_gcn.equi_conv2(self.dim_list[3], self.dim_list[4])
        else:
            self.conv_1 = equi_gcn.equi_conv(self.dim_list[0], self.dim_list[1])
            self.pool_1 = equi_gcn.equ_pool_layer2(pooling_rate= 4, neighbor_num= 4)
            self.conv_2 = equi_gcn.equi_conv(self.dim_list[1], self.dim_list[2])
            self.conv_3 = equi_gcn.equi_conv(self.dim_list[2], self.dim_list[3])
            self.pool_2 = equi_gcn.equ_pool_layer2(pooling_rate= 4, neighbor_num= 4)
            self.conv_4 = equi_gcn.equi_conv(self.dim_list[3], self.dim_list[4])

        self.bn1 = nn.BatchNorm2d(self.dim_list[1])
        self.bn2 = nn.BatchNorm2d(self.dim_list[2])
        self.bn3 = nn.BatchNorm2d(self.dim_list[3])

    def forward(self,vertices: "tensor (bs, vetice_num, 3)"):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace= True)
        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0)), inplace= True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1)), inplace= True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2)), inplace= True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_3num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        #
        feature_dict={'conv0':{'pos':vertices,'fea':torch.max(fm_0,dim=-1)[0].transpose(1,2)},
                      'conv1':{'pos':vertices,'fea':torch.max(fm_1,dim=-1)[0].transpose(1,2)},
                      'conv2':{'pos':v_pool_1,'fea':torch.max(fm_2,dim=-1)[0].transpose(1,2)},
                      'conv3':{'pos':v_pool_1,'fea':torch.max(fm_3,dim=-1)[0].transpose(1,2)},
                      'conv4':{'pos':v_pool_2,'fea':torch.max(fm_4,dim=-1)[0].transpose(1,2)},

                      }

        return feature_dict


class Equi_gcn2(nn.Module):
    def __init__(self, neighbor_num):
        super(Equi_gcn2, self).__init__()
        self.neighbor_num = neighbor_num
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        self.conv_0 = equi_gcn.equi_conv4(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
        self.conv_1 = equi_gcn.equi_conv4(self.dim_list[0], self.dim_list[1])
        self.pool_1 = equi_gcn.equ_pool_layer3(self.dim_list[1],pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = equi_gcn.equi_conv4(self.dim_list[1], self.dim_list[2])
        self.conv_3 = equi_gcn.equi_conv4(self.dim_list[2], self.dim_list[3])
        self.pool_2 = equi_gcn.equ_pool_layer3(self.dim_list[3],pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = equi_gcn.equi_conv4(self.dim_list[3], self.dim_list[4])

        self.anchor_pool_0=tovec(self.dim_list[0],self.dim_list[0])
        self.anchor_pool_1=tovec(self.dim_list[1],self.dim_list[1])
        self.anchor_pool_2=tovec(self.dim_list[2],self.dim_list[2])
        self.anchor_pool_3=tovec(self.dim_list[3],self.dim_list[3])
        self.anchor_pool_4=tovec(self.dim_list[4],self.dim_list[4])


        self.bn1 = nn.BatchNorm2d(self.dim_list[1])
        self.bn2 = nn.BatchNorm2d(self.dim_list[2])
        self.bn3 = nn.BatchNorm2d(self.dim_list[3])

        self.conv1d_block = nn.Sequential(
            VNLinear(sum(self.dim_list), 256),
            VNReLU(256),
            VNLinear(256, 256),
            VNReLU(256),
            VNLinear(256, 1),
        )


    # self.tovec=tovec(self.dim_list[0])


    def forward(self,vertices,return_fuse=False):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = equi_gcn.get_neighbor_index(vertices, self.neighbor_num)


        fm_0 = F.relu(self.conv_0(neighbor_index, vertices)[0], inplace= True)
        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0)[0]), inplace= True)

        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)

        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1)[0]), inplace= True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2)[0]), inplace= True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_3num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                v_pool_2.shape[1] // 8))
        # _, neighbor_index, dst_nn_in_src = knn_points(
        #     v_pool_2, v_pool_2, K=min(self.neighbor_num,v_pool_2.shape[1] // 8), return_nn=True)
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)[0]

        # return fm_4
        # print(fm_4 [:,4,0,:])
        feature_dict={'conv0':{'pos':vertices,'fea':self.anchor_pool_0(fm_0,self.conv_0.vs).transpose(1,2)},
                      'conv1':{'pos':vertices,'fea':self.anchor_pool_1(fm_1,self.conv_0.vs).transpose(1,2)},
                      'conv2':{'pos':v_pool_1,'fea':self.anchor_pool_2(fm_2,self.conv_0.vs).transpose(1,2)},
                      'conv3':{'pos':v_pool_1,'fea':self.anchor_pool_3(fm_3,self.conv_0.vs).transpose(1,2)},
                      'conv4':{'pos':v_pool_2,'fea':self.anchor_pool_4(fm_4,self.conv_0.vs).transpose(1,2)},
                      }
        # print(feature_dict)
        if return_fuse:
            # nearest_pool_1 = equi_gcn.get_nearest_index(vertices, v_pool_1)
            # nearest_pool_2 = equi_gcn.get_nearest_index(vertices, v_pool_2)
            # fm_2 = equi_gcn.indexing_neighbor(fm_2.permute(0,2,3,1), nearest_pool_1).squeeze(2).permute(0,3,1,2)
            # fm_3 = equi_gcn.indexing_neighbor(fm_3.permute(0,2,3,1), nearest_pool_1).squeeze(2).permute(0,3,1,2)
            # fm_4 = equi_gcn.indexing_neighbor(fm_4.permute(0,2,3,1), nearest_pool_2).squeeze(2).permute(0,3,1,2)
            # fm_fuse = torch.cat([self.anchor_pool_0(fm_0,self.conv_0.vs),
            #                      self.anchor_pool_1(fm_1,self.conv_0.vs),
            #                      self.anchor_pool_2(fm_2,self.conv_0.vs),
            #                      self.anchor_pool_3(fm_3,self.conv_0.vs),
            #                      self.anchor_pool_4(fm_4,self.conv_0.vs)], dim= 1)
            # fm_fuse=fm_fuse.permute(0,2,1,3)
            # fuse_nocs=self.conv1d_block(fm_fuse)
            # return feature_dict,fm_fuse,fuse_nocs.squeeze(2)
            fuse=torch.max(fm_4,dim=2,keepdim=True)[0]
            fuse=self.anchor_pool_4(fuse,self.conv_0.vs).transpose(1,2)
            fuse=self.conv1d_block(fuse).squeeze()
            return feature_dict,fuse
        else:
            return feature_dict



class Equi_gcn3(nn.Module):
    def __init__(self, neighbor_num):
        super(Equi_gcn3, self).__init__()
        self.neighbor_num = neighbor_num
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        self.conv_0 = equi_gcn.equi_conv4(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
        self.conv_1 = equi_gcn.equi_conv4(self.dim_list[0], self.dim_list[1])
        self.pool_1 = equi_gcn.equ_pool_layer3(self.dim_list[1],pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = equi_gcn.equi_conv4(self.dim_list[1], self.dim_list[2])
        self.conv_3 = equi_gcn.equi_conv4(self.dim_list[2], self.dim_list[3])
        self.pool_2 = equi_gcn.equ_pool_layer3(self.dim_list[3],pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = equi_gcn.equi_conv4(self.dim_list[3], self.dim_list[4])


        # self.anchor_pool_4=tovec(self.dim_list[4],self.dim_list[4])
        if FLAGS.tovec_version=='v2':
            self.anchor_pool_4=tovec_2(self.dim_list[4])
        elif FLAGS.tovec_version=='v1':
            self.anchor_pool_4=tovec(self.dim_list[4])
        elif FLAGS.tovec_version=='v3':
            self.anchor_pool_4=tovec_3(self.dim_list[4])

        self.bn1 = nn.BatchNorm1d(self.dim_list[1]*12)
        self.bn2 = nn.BatchNorm1d(self.dim_list[2]*12)
        self.bn3 = nn.BatchNorm1d(self.dim_list[3]*12)
        self.bn4 = nn.BatchNorm1d(self.dim_list[4]*12)

        self.conv1d_block = nn.Sequential(
            VNLinear(sum(self.dim_list), 256),
            VNReLU(256),
            VNLinear(256, 256),
            VNReLU(256),
            VNLinear(256, 1),
        )


    # self.tovec=tovec(self.dim_list[0])


    def forward(self,vertices,return_fuse=False):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = equi_gcn.get_neighbor_index(vertices, self.neighbor_num)


        fm_0 = F.relu(self.conv_0(neighbor_index, vertices)[0], inplace= True)
        tmp_1,fm_0_dir=self.conv_1(neighbor_index, vertices, fm_0)
        cur_num=tmp_1.shape[2]
        tmp_1=tmp_1.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_1 = F.relu(self.bn1(tmp_1), inplace= True)
        fm_1 = fm_1.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)

        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_1,
                                                     min(self.neighbor_num, v_pool_1.shape[1] // 8))
        tmp_2,fm_1_dir=self.conv_2(neighbor_index, v_pool_1, fm_pool_1)
        cur_num=tmp_2.shape[2]
        tmp_2=tmp_2.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_2 = F.relu(self.bn2(tmp_2), inplace= True)
        fm_2 = fm_2.reshape(bs,-1,12,cur_num).permute(0,1,3,2)


        tmp_3,fm_2_dir=self.conv_3(neighbor_index, v_pool_1, fm_2)
        cur_num=tmp_3.shape[2]
        tmp_3=tmp_3.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_3= F.relu(self.bn3(tmp_3), inplace= True)
        fm_3 = fm_3.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_3num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                   v_pool_2.shape[1] // 8))
        # _, neighbor_index, dst_nn_in_src = knn_points(
        #     v_pool_2, v_pool_2, K=min(self.neighbor_num,v_pool_2.shape[1] // 8), return_nn=True)
        tmp_4,fm_3_dir= self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        cur_num=tmp_4.shape[2]
        tmp_4=tmp_4.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_4 = F.relu(self.bn4(tmp_4), inplace= True)
        fm_4 = fm_4.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        # return fm_4
        # print(fm_4 [:,4,0,:])
        feature_dict={'conv0':{'pos':vertices,'fea':fm_0_dir.transpose(1,2)},
                      'conv1':{'pos':v_pool_1,'fea':fm_1_dir.transpose(1,2)},
                      'conv2':{'pos':v_pool_1,'fea':fm_2_dir.transpose(1,2)},
                      'conv3':{'pos':v_pool_2,'fea':fm_3_dir.transpose(1,2)},
                      'conv4':{'pos':v_pool_2,'fea':self.anchor_pool_4(fm_4,self.conv_0.vs).transpose(1,2)},
                      }
        # print(feature_dict)
        if return_fuse:
            # nearest_pool_1 = equi_gcn.get_nearest_index(vertices, v_pool_1)
            # nearest_pool_2 = equi_gcn.get_nearest_index(vertices, v_pool_2)
            # fm_2 = equi_gcn.indexing_neighbor(fm_2.permute(0,2,3,1), nearest_pool_1).squeeze(2).permute(0,3,1,2)
            # fm_3 = equi_gcn.indexing_neighbor(fm_3.permute(0,2,3,1), nearest_pool_1).squeeze(2).permute(0,3,1,2)
            # fm_4 = equi_gcn.indexing_neighbor(fm_4.permute(0,2,3,1), nearest_pool_2).squeeze(2).permute(0,3,1,2)
            # fm_fuse = torch.cat([self.anchor_pool_0(fm_0,self.conv_0.vs),
            #                      self.anchor_pool_1(fm_1,self.conv_0.vs),
            #                      self.anchor_pool_2(fm_2,self.conv_0.vs),
            #                      self.anchor_pool_3(fm_3,self.conv_0.vs),
            #                      self.anchor_pool_4(fm_4,self.conv_0.vs)], dim= 1)
            # fm_fuse=fm_fuse.permute(0,2,1,3)
            # fuse_nocs=self.conv1d_block(fm_fuse)
            # return feature_dict,fm_fuse,fuse_nocs.squeeze(2)
            fuse=torch.max(fm_4,dim=2,keepdim=True)[0]
            fuse=self.anchor_pool_4(fuse,self.conv_0.vs).transpose(1,2)
            fuse=self.conv1d_block(fuse).squeeze()
            return feature_dict,fuse
        else:
            return feature_dict




class Equi_gcn4(nn.Module):
    def __init__(self, neighbor_num):
        super(Equi_gcn4, self).__init__()
        self.neighbor_num = neighbor_num
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        self.conv_0 = equi_gcn.equi_conv5(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
        self.conv_1 = equi_gcn.equi_conv5(self.dim_list[0], self.dim_list[1])
        self.pool_1 = equi_gcn.equ_pool_layer3(self.dim_list[1],pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = equi_gcn.equi_conv5(self.dim_list[1], self.dim_list[2])
        self.conv_3 = equi_gcn.equi_conv5(self.dim_list[2], self.dim_list[3])
        self.pool_2 = equi_gcn.equ_pool_layer3(self.dim_list[3],pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = equi_gcn.equi_conv5(self.dim_list[3], self.dim_list[4])


        # self.anchor_pool_4=tovec(self.dim_list[4],self.dim_list[4])
        if FLAGS.tovec_version=='v2':
            self.anchor_pool_4=tovec_2(self.dim_list[4])
        elif FLAGS.tovec_version=='v1':
            self.anchor_pool_4=tovec(self.dim_list[4])
        elif FLAGS.tovec_version=='v3':
            self.anchor_pool_4=tovec_3(self.dim_list[4])

        self.bn1 = nn.BatchNorm1d(self.dim_list[1]*12)
        self.bn2 = nn.BatchNorm1d(self.dim_list[2]*12)
        self.bn3 = nn.BatchNorm1d(self.dim_list[3]*12)
        self.bn4 = nn.BatchNorm1d(self.dim_list[4]*12)

        self.conv1d_block = nn.Sequential(
            VNLinear(sum(self.dim_list), 256),
            VNReLU(256),
            VNLinear(256, 256),
            VNReLU(256),
            VNLinear(256, 1),
        )


    # self.tovec=tovec(self.dim_list[0])


    def forward(self,vertices,return_fuse=False):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = equi_gcn.get_neighbor_index(vertices, self.neighbor_num)


        fm_0 = F.relu(self.conv_0(neighbor_index, vertices)[0], inplace= True)
        tmp_1,fm_0_dir=self.conv_1(neighbor_index, vertices, fm_0)
        cur_num=tmp_1.shape[2]
        tmp_1=tmp_1.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_1 = F.relu(self.bn1(tmp_1), inplace= True)
        fm_1 = fm_1.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)

        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_1,
                                                     min(self.neighbor_num, v_pool_1.shape[1] // 8))
        tmp_2,fm_1_dir=self.conv_2(neighbor_index, v_pool_1, fm_pool_1)
        cur_num=tmp_2.shape[2]
        tmp_2=tmp_2.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_2 = F.relu(self.bn2(tmp_2), inplace= True)
        fm_2 = fm_2.reshape(bs,-1,12,cur_num).permute(0,1,3,2)


        tmp_3,fm_2_dir=self.conv_3(neighbor_index, v_pool_1, fm_2)
        cur_num=tmp_3.shape[2]
        tmp_3=tmp_3.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_3= F.relu(self.bn3(tmp_3), inplace= True)
        fm_3 = fm_3.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_3num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                   v_pool_2.shape[1] // 8))
        # _, neighbor_index, dst_nn_in_src = knn_points(
        #     v_pool_2, v_pool_2, K=min(self.neighbor_num,v_pool_2.shape[1] // 8), return_nn=True)
        tmp_4,fm_3_dir= self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        cur_num=tmp_4.shape[2]
        tmp_4=tmp_4.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_4 = F.relu(self.bn4(tmp_4), inplace= True)
        fm_4 = fm_4.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        # return fm_4
        # print(fm_4 [:,4,0,:])
        feature_dict={'conv0':{'pos':vertices,'fea':fm_0_dir.transpose(1,2)},
                      'conv1':{'pos':v_pool_1,'fea':fm_1_dir.transpose(1,2)},
                      'conv2':{'pos':v_pool_1,'fea':fm_2_dir.transpose(1,2)},
                      'conv3':{'pos':v_pool_2,'fea':fm_3_dir.transpose(1,2)},
                      'conv4':{'pos':v_pool_2,'fea':self.anchor_pool_4(fm_4,self.conv_0.vs).transpose(1,2)},
                      }
        # print(feature_dict)
        if return_fuse:
            # nearest_pool_1 = equi_gcn.get_nearest_index(vertices, v_pool_1)
            # nearest_pool_2 = equi_gcn.get_nearest_index(vertices, v_pool_2)
            # fm_2 = equi_gcn.indexing_neighbor(fm_2.permute(0,2,3,1), nearest_pool_1).squeeze(2).permute(0,3,1,2)
            # fm_3 = equi_gcn.indexing_neighbor(fm_3.permute(0,2,3,1), nearest_pool_1).squeeze(2).permute(0,3,1,2)
            # fm_4 = equi_gcn.indexing_neighbor(fm_4.permute(0,2,3,1), nearest_pool_2).squeeze(2).permute(0,3,1,2)
            # fm_fuse = torch.cat([self.anchor_pool_0(fm_0,self.conv_0.vs),
            #                      self.anchor_pool_1(fm_1,self.conv_0.vs),
            #                      self.anchor_pool_2(fm_2,self.conv_0.vs),
            #                      self.anchor_pool_3(fm_3,self.conv_0.vs),
            #                      self.anchor_pool_4(fm_4,self.conv_0.vs)], dim= 1)
            # fm_fuse=fm_fuse.permute(0,2,1,3)
            # fuse_nocs=self.conv1d_block(fm_fuse)
            # return feature_dict,fm_fuse,fuse_nocs.squeeze(2)
            fuse=torch.max(fm_4,dim=2,keepdim=True)[0]
            fuse=self.anchor_pool_4(fuse,self.conv_0.vs).transpose(1,2)
            fuse=self.conv1d_block(fuse).squeeze()
            return feature_dict,fuse
        else:
            return feature_dict


class Equi_gcn5(nn.Module):
    def __init__(self, neighbor_num):
        super(Equi_gcn5, self).__init__()
        self.neighbor_num = neighbor_num
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        self.conv_0 = equi_gcn.equi_conv6(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
        self.conv_1 = equi_gcn.equi_conv6(self.dim_list[0], self.dim_list[1])
        self.pool_1 = equi_gcn.equ_pool_layer3(self.dim_list[1],pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = equi_gcn.equi_conv6(self.dim_list[1], self.dim_list[2])
        self.conv_3 = equi_gcn.equi_conv6(self.dim_list[2], self.dim_list[3])
        self.pool_2 = equi_gcn.equ_pool_layer3(self.dim_list[3],pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = equi_gcn.equi_conv6(self.dim_list[3], self.dim_list[4])


        # self.anchor_pool_4=tovec(self.dim_list[4],self.dim_list[4])
        if FLAGS.tovec_version=='v2':
            self.anchor_pool_4=tovec_2(self.dim_list[4])
        elif FLAGS.tovec_version=='v1':
            self.anchor_pool_4=tovec(self.dim_list[4])
        elif FLAGS.tovec_version=='v3':
            self.anchor_pool_4=tovec_3(self.dim_list[4])

        self.bn1 = nn.BatchNorm1d(self.dim_list[1]*12)
        self.bn2 = nn.BatchNorm1d(self.dim_list[2]*12)
        self.bn3 = nn.BatchNorm1d(self.dim_list[3]*12)
        self.bn4 = nn.BatchNorm1d(self.dim_list[4]*12)

        self.conv1d_block = nn.Sequential(
            VNLinear(sum(self.dim_list), 256),
            VNReLU(256),
            VNLinear(256, 256),
            VNReLU(256),
            VNLinear(256, 1),
        )


    # self.tovec=tovec(self.dim_list[0])


    def forward(self,vertices,return_fuse=False):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = equi_gcn.get_neighbor_index(vertices, self.neighbor_num)


        fm_0 = F.relu(self.conv_0(neighbor_index, vertices)[0], inplace= True)
        tmp_1,fm_0_dir=self.conv_1(neighbor_index, vertices, fm_0)
        cur_num=tmp_1.shape[2]
        tmp_1=tmp_1.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_1 = F.relu(self.bn1(tmp_1), inplace= True)
        fm_1 = fm_1.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)

        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_1,
                                                     min(self.neighbor_num, v_pool_1.shape[1] // 8))
        tmp_2,fm_1_dir=self.conv_2(neighbor_index, v_pool_1, fm_pool_1)
        cur_num=tmp_2.shape[2]
        tmp_2=tmp_2.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_2 = F.relu(self.bn2(tmp_2), inplace= True)
        fm_2 = fm_2.reshape(bs,-1,12,cur_num).permute(0,1,3,2)


        tmp_3,fm_2_dir=self.conv_3(neighbor_index, v_pool_1, fm_2)
        cur_num=tmp_3.shape[2]
        tmp_3=tmp_3.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_3= F.relu(self.bn3(tmp_3), inplace= True)
        fm_3 = fm_3.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_3num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                   v_pool_2.shape[1] // 8))
        # _, neighbor_index, dst_nn_in_src = knn_points(
        #     v_pool_2, v_pool_2, K=min(self.neighbor_num,v_pool_2.shape[1] // 8), return_nn=True)
        tmp_4,fm_3_dir= self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        cur_num=tmp_4.shape[2]
        tmp_4=tmp_4.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_4 = F.relu(self.bn4(tmp_4), inplace= True)
        fm_4 = fm_4.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        # return fm_4
        # print(fm_4 [:,4,0,:])
        feature_dict={'conv0':{'pos':vertices,'fea':fm_0_dir.transpose(1,2)},
                      'conv1':{'pos':v_pool_1,'fea':fm_1_dir.transpose(1,2)},
                      'conv2':{'pos':v_pool_1,'fea':fm_2_dir.transpose(1,2)},
                      'conv3':{'pos':v_pool_2,'fea':fm_3_dir.transpose(1,2)},
                      'conv4':{'pos':v_pool_2,'fea':self.anchor_pool_4(fm_4,self.conv_0.vs).transpose(1,2)},
                      }
        # print(feature_dict)
        if return_fuse:
            # nearest_pool_1 = equi_gcn.get_nearest_index(vertices, v_pool_1)
            # nearest_pool_2 = equi_gcn.get_nearest_index(vertices, v_pool_2)
            # fm_2 = equi_gcn.indexing_neighbor(fm_2.permute(0,2,3,1), nearest_pool_1).squeeze(2).permute(0,3,1,2)
            # fm_3 = equi_gcn.indexing_neighbor(fm_3.permute(0,2,3,1), nearest_pool_1).squeeze(2).permute(0,3,1,2)
            # fm_4 = equi_gcn.indexing_neighbor(fm_4.permute(0,2,3,1), nearest_pool_2).squeeze(2).permute(0,3,1,2)
            # fm_fuse = torch.cat([self.anchor_pool_0(fm_0,self.conv_0.vs),
            #                      self.anchor_pool_1(fm_1,self.conv_0.vs),
            #                      self.anchor_pool_2(fm_2,self.conv_0.vs),
            #                      self.anchor_pool_3(fm_3,self.conv_0.vs),
            #                      self.anchor_pool_4(fm_4,self.conv_0.vs)], dim= 1)
            # fm_fuse=fm_fuse.permute(0,2,1,3)
            # fuse_nocs=self.conv1d_block(fm_fuse)
            # return feature_dict,fm_fuse,fuse_nocs.squeeze(2)
            fuse=torch.max(fm_4,dim=2,keepdim=True)[0]
            fuse=self.anchor_pool_4(fuse,self.conv_0.vs).transpose(1,2)
            fuse=self.conv1d_block(fuse).squeeze()
            return feature_dict,fuse
        else:
            return feature_dict


class Equi_diff_gcn(nn.Module):
    def __init__(self, neighbor_num):
        super(Equi_diff_gcn, self).__init__()
        self.neighbor_num = neighbor_num
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        self.conv_0 = equi_gcn.equi_conv5(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
        self.conv_1 = equi_gcn.equi_conv5(self.dim_list[0], self.dim_list[1])
        self.pool_1 = equi_gcn.equ_pool_layer3(self.dim_list[1],pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = equi_gcn.equi_conv5(self.dim_list[1], self.dim_list[2])
        self.conv_3 = equi_gcn.equi_conv5(self.dim_list[2], self.dim_list[3])
        self.pool_2 = equi_gcn.equ_pool_layer3(self.dim_list[3],pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = equi_gcn.equi_conv5(self.dim_list[3], self.dim_list[4])


        # self.anchor_pool_4=tovec(self.dim_list[4],self.dim_list[4])
        if FLAGS.tovec_version=='v2':
            self.anchor_pool_4=tovec_2(self.dim_list[4])
        elif FLAGS.tovec_version=='v1':
            self.anchor_pool_4=tovec(self.dim_list[4])
        elif FLAGS.tovec_version=='v3':
            self.anchor_pool_4=tovec_3(self.dim_list[4])

        self.bn1 = nn.BatchNorm1d(self.dim_list[1]*12)
        self.bn2 = nn.BatchNorm1d(self.dim_list[2]*12)
        self.bn3 = nn.BatchNorm1d(self.dim_list[3]*12)
        self.bn4 = nn.BatchNorm1d(self.dim_list[4]*12)

        self.conv1d_block = nn.Sequential(
            VNLinear(sum(self.dim_list), 256),
            VNReLU(256),
            VNLinear(256, 256),
            VNReLU(256),
            VNLinear(256, 1),
        )


    # self.tovec=tovec(self.dim_list[0])


    def forward(self,vertices,return_fuse=False):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = equi_gcn.get_neighbor_index(vertices, self.neighbor_num)


        fm_0 = F.relu(self.conv_0(neighbor_index, vertices)[0], inplace= True)
        tmp_1,fm_0_dir=self.conv_1(neighbor_index, vertices, fm_0)
        cur_num=tmp_1.shape[2]
        tmp_1=tmp_1.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_1 = F.relu(self.bn1(tmp_1), inplace= True)
        fm_1 = fm_1.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)

        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_1,
                                                     min(self.neighbor_num, v_pool_1.shape[1] // 8))
        tmp_2,fm_1_dir=self.conv_2(neighbor_index, v_pool_1, fm_pool_1)
        cur_num=tmp_2.shape[2]
        tmp_2=tmp_2.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_2 = F.relu(self.bn2(tmp_2), inplace= True)
        fm_2 = fm_2.reshape(bs,-1,12,cur_num).permute(0,1,3,2)


        tmp_3,fm_2_dir=self.conv_3(neighbor_index, v_pool_1, fm_2)
        cur_num=tmp_3.shape[2]
        tmp_3=tmp_3.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_3= F.relu(self.bn3(tmp_3), inplace= True)
        fm_3 = fm_3.reshape(bs,-1,12,cur_num).permute(0,1,3,2)

        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_3num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                   v_pool_2.shape[1] // 8))
        # _, neighbor_index, dst_nn_in_src = knn_points(
        #     v_pool_2, v_pool_2, K=min(self.neighbor_num,v_pool_2.shape[1] // 8), return_nn=True)
        tmp_4,fm_3_dir= self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        cur_num=tmp_4.shape[2]
        tmp_4=tmp_4.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_4 = F.relu(self.bn4(tmp_4), inplace= True)

        final_fea= fm_4.reshape(bs,-1,12,cur_num).permute(0,3,2,1)
        if not return_fuse:
            fm_4     = fm_4.reshape(bs,-1,12,cur_num).permute(0,1,3,2)
            feature_dict={'conv0':{'pos':vertices,'fea':fm_0_dir.transpose(1,2)},
                          'conv1':{'pos':v_pool_1,'fea':fm_1_dir.transpose(1,2)},
                          'conv2':{'pos':v_pool_1,'fea':fm_2_dir.transpose(1,2)},
                          'conv3':{'pos':v_pool_2,'fea':fm_3_dir.transpose(1,2)},
                          'conv4':{'pos':v_pool_2,'fea':self.anchor_pool_4(fm_4,self.conv_0.vs).transpose(1,2)},
                          }
            return None,None,feature_dict
        else:
            fm_4= fm_4.reshape(bs,-1,12,cur_num).permute(0,3,2,1)
            final_point=v_pool_2
            if FLAGS.use_fuse and not FLAGS.use_simple:
                nearest_pool_0 = equi_gcn.get_nearest_index(v_pool_2, vertices)
                nearest_pool_1 = equi_gcn.get_nearest_index(v_pool_2, v_pool_1)

                fm_1 = equi_gcn.indexing_neighbor(fm_1.permute(0,2,3,1), nearest_pool_0).squeeze(2)
                fm_2 = equi_gcn.indexing_neighbor(fm_2.permute(0,2,3,1), nearest_pool_1).squeeze(2)
                fm_3 = equi_gcn.indexing_neighbor(fm_3.permute(0,2,3,1), nearest_pool_1).squeeze(2)
                final_fea=torch.cat([fm_1,fm_2,fm_3,fm_4],dim=-1)
            else:
                final_fea=fm_4
            return final_fea,final_point,None



class Equi_diff_gcn_a5(nn.Module):
    def __init__(self, neighbor_num):
        super(Equi_diff_gcn_a5, self).__init__()
        self.neighbor_num = neighbor_num
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        self.conv_0 = equi_gcn.equi_conv7(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
        self.conv_1 = equi_gcn.equi_conv7(self.dim_list[0], self.dim_list[1])
        self.pool_1 = equi_gcn.equ_pool_layer3(self.dim_list[1],pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = equi_gcn.equi_conv7(self.dim_list[1], self.dim_list[2])
        self.conv_3 = equi_gcn.equi_conv7(self.dim_list[2], self.dim_list[3])
        self.pool_2 = equi_gcn.equ_pool_layer3(self.dim_list[3],pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = equi_gcn.equi_conv7(self.dim_list[3], self.dim_list[4])





        self.bn0 = nn.BatchNorm1d(self.dim_list[0]*5)
        self.bn1 = nn.BatchNorm1d(self.dim_list[1]*5)
        self.bn2 = nn.BatchNorm1d(self.dim_list[2]*5)
        self.bn3 = nn.BatchNorm1d(self.dim_list[3]*5)
        self.bn4 = nn.BatchNorm1d(self.dim_list[4]*5)

        self.conv1d_block = nn.Sequential(
            VNLinear(sum(self.dim_list), 256),
            VNReLU(256),
            VNLinear(256, 256),
            VNReLU(256),
            VNLinear(256, 1),
        )


    # self.tovec=tovec(self.dim_list[0])


    def forward(self,vertices,return_fuse=False):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = equi_gcn.get_neighbor_index(vertices, self.neighbor_num)

        tmp_0=self.conv_0(neighbor_index, vertices)[0]
        cur_num=tmp_0.shape[2]
        tmp_0=tmp_0.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_0=F.relu(self.bn0(tmp_0), inplace= True)
        fm_0=fm_0.reshape(bs,-1,5,cur_num).permute(0,1,3,2)
        tmp_1,fm_0_dir=self.conv_1(neighbor_index, vertices, fm_0)
        cur_num=tmp_1.shape[2]
        tmp_1=tmp_1.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_1 = F.relu(self.bn1(tmp_1), inplace= True)
        fm_1 = fm_1.reshape(bs,-1,5,cur_num).permute(0,1,3,2)

        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)

        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_1,
                                                     min(self.neighbor_num, v_pool_1.shape[1] // 8))
        tmp_2,fm_1_dir=self.conv_2(neighbor_index, v_pool_1, fm_pool_1)
        cur_num=tmp_2.shape[2]
        tmp_2=tmp_2.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_2 = F.relu(self.bn2(tmp_2), inplace= True)
        fm_2 = fm_2.reshape(bs,-1,5,cur_num).permute(0,1,3,2)


        tmp_3,fm_2_dir=self.conv_3(neighbor_index, v_pool_1, fm_2)
        cur_num=tmp_3.shape[2]
        tmp_3=tmp_3.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_3= F.relu(self.bn3(tmp_3), inplace= True)
        fm_3 = fm_3.reshape(bs,-1,5,cur_num).permute(0,1,3,2)

        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_3num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                   v_pool_2.shape[1] // 8))
        # _, neighbor_index, dst_nn_in_src = knn_points(
        #     v_pool_2, v_pool_2, K=min(self.neighbor_num,v_pool_2.shape[1] // 8), return_nn=True)
        tmp_4,fm_3_dir= self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        cur_num=tmp_4.shape[2]
        tmp_4=tmp_4.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_4 = F.relu(self.bn4(tmp_4), inplace= True)

        final_fea= fm_4.reshape(bs,-1,5,cur_num).permute(0,3,2,1)

        if not return_fuse:
            fm_4     = fm_4.reshape(bs,-1,5,cur_num).permute(0,1,3,2)
            feature_dict={'conv0':{'pos':vertices,'fea':fm_0_dir.transpose(1,2)},
                      'conv1':{'pos':v_pool_1,'fea':fm_1_dir.transpose(1,2)},
                      'conv2':{'pos':v_pool_1,'fea':fm_2_dir.transpose(1,2)},
                      'conv3':{'pos':v_pool_2,'fea':fm_3_dir.transpose(1,2)},
                      'conv4':{'pos':v_pool_2,'fea':fm_4.transpose(1,2)},
                      }
            return None,None,feature_dict
        else:
            fm_4= fm_4.reshape(bs,-1,5,cur_num).permute(0,3,2,1)
            final_point=v_pool_2
            if FLAGS.use_fuse:
                nearest_pool_0 = equi_gcn.get_nearest_index(v_pool_2, vertices)
                nearest_pool_1 = equi_gcn.get_nearest_index(v_pool_2, v_pool_1)

                fm_1 = equi_gcn.indexing_neighbor(fm_1.permute(0,2,3,1), nearest_pool_0).squeeze(2)
                fm_2 = equi_gcn.indexing_neighbor(fm_2.permute(0,2,3,1), nearest_pool_1).squeeze(2)
                fm_3 = equi_gcn.indexing_neighbor(fm_3.permute(0,2,3,1), nearest_pool_1).squeeze(2)
                final_fea=torch.cat([fm_1,fm_2,fm_3,fm_4],dim=-1)
            else:
                final_fea=fm_4
            return final_fea,final_point,None

        # return fm_4



class Equi_diff_gcn_a6(nn.Module):
    def __init__(self, neighbor_num):
        super(Equi_diff_gcn_a6, self).__init__()
        self.neighbor_num = neighbor_num
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        self.conv_0 = equi_gcn.equi_conv8(dim_in=1,dim_out=self.dim_list[0],is_surface=True)
        self.conv_1 = equi_gcn.equi_conv8(self.dim_list[0], self.dim_list[1])
        self.pool_1 = equi_gcn.equ_pool_layer3(self.dim_list[1],pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = equi_gcn.equi_conv8(self.dim_list[1], self.dim_list[2])
        self.conv_3 = equi_gcn.equi_conv8(self.dim_list[2], self.dim_list[3])
        self.pool_2 = equi_gcn.equ_pool_layer3(self.dim_list[3],pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = equi_gcn.equi_conv8(self.dim_list[3], self.dim_list[4])





        self.bn0 = nn.BatchNorm1d(self.dim_list[0]*6)
        self.bn1 = nn.BatchNorm1d(self.dim_list[1]*6)
        self.bn2 = nn.BatchNorm1d(self.dim_list[2]*6)
        self.bn3 = nn.BatchNorm1d(self.dim_list[3]*6)
        self.bn4 = nn.BatchNorm1d(self.dim_list[4]*6)

        self.conv1d_block = nn.Sequential(
            VNLinear(sum(self.dim_list), 256),
            VNReLU(256),
            VNLinear(256, 256),
            VNReLU(256),
            VNLinear(256, 1),
        )


    # self.tovec=tovec(self.dim_list[0])


    def forward(self,vertices,return_fuse=False):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = equi_gcn.get_neighbor_index(vertices, self.neighbor_num)

        tmp_0=self.conv_0(neighbor_index, vertices)[0]
        cur_num=tmp_0.shape[2]
        tmp_0=tmp_0.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_0=F.relu(self.bn0(tmp_0), inplace= True)
        fm_0=fm_0.reshape(bs,-1,6,cur_num).permute(0,1,3,2)
        tmp_1,fm_0_dir=self.conv_1(neighbor_index, vertices, fm_0)
        cur_num=tmp_1.shape[2]
        tmp_1=tmp_1.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_1 = F.relu(self.bn1(tmp_1), inplace= True)
        fm_1 = fm_1.reshape(bs,-1,6,cur_num).permute(0,1,3,2)

        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)

        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_1,
                                                     min(self.neighbor_num, v_pool_1.shape[1] // 8))
        tmp_2,fm_1_dir=self.conv_2(neighbor_index, v_pool_1, fm_pool_1)
        cur_num=tmp_2.shape[2]
        tmp_2=tmp_2.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_2 = F.relu(self.bn2(tmp_2), inplace= True)
        fm_2 = fm_2.reshape(bs,-1,6,cur_num).permute(0,1,3,2)


        tmp_3,fm_2_dir=self.conv_3(neighbor_index, v_pool_1, fm_2)
        cur_num=tmp_3.shape[2]
        tmp_3=tmp_3.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_3= F.relu(self.bn3(tmp_3), inplace= True)
        fm_3 = fm_3.reshape(bs,-1,6,cur_num).permute(0,1,3,2)

        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_3num)
        neighbor_index = equi_gcn.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                   v_pool_2.shape[1] // 8))
        # _, neighbor_index, dst_nn_in_src = knn_points(
        #     v_pool_2, v_pool_2, K=min(self.neighbor_num,v_pool_2.shape[1] // 8), return_nn=True)
        tmp_4,fm_3_dir= self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        cur_num=tmp_4.shape[2]
        tmp_4=tmp_4.permute(0,1,3,2).reshape(bs,-1,cur_num)
        fm_4 = F.relu(self.bn4(tmp_4), inplace= True)

        final_fea= fm_4.reshape(bs,-1,6,cur_num).permute(0,3,2,1)
        fm_4     = fm_4.reshape(bs,-1,6,cur_num).permute(0,1,3,2)
        feature_dict={'conv0':{'pos':vertices,'fea':fm_0_dir.transpose(1,2)},
                      'conv1':{'pos':v_pool_1,'fea':fm_1_dir.transpose(1,2)},
                      'conv2':{'pos':v_pool_1,'fea':fm_2_dir.transpose(1,2)},
                      'conv3':{'pos':v_pool_2,'fea':fm_3_dir.transpose(1,2)},
                      'conv4':{'pos':v_pool_2,'fea':fm_4.transpose(1,2)},
                      }
        final_point=v_pool_2
        return final_fea,final_point,feature_dict

        # return fm_4







def seed_everything(seed=20):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
if __name__ == "__main__":


    import os
    # os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    torch.backends.cuda.matmul.allow_tf32=False
    seed_everything()
    def debug(argv):
        if FLAGS.debug_path is not '':
            debug_dict=torch.load(FLAGS.debug_path)
            vertices=debug_dict['vertices']
        else:
            vertices = torch.randn(4, 1024, 3)
            debug_dict={}
            debug_dict['vertices']=vertices
            torch.save(debug_dict, FLAGS.debug_path)
        vertices=vertices.cuda()
        vertices_=vertices.clone()
        network = NFPose().cuda().eval()
        pred_scale=torch.ones(vertices.shape[0])*0.5
        pred_scale=pred_scale.float().cuda()
        cat_name='laptop'
        resume_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
        network.load_state_dict(torch.load(resume_path))
        out=network.backbone1(vertices)

        out=network.qnet(vertices,out,pred_scale)
        print(out)


    def eval_Equi_gcn2(argv):
        vertices = torch.randn(1, 200, 3).cuda()
        R12=get_anchorsV12()
        R60=get_anchorsV()



        index=23
        index2=45



        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()
        conv=Equi_gcn2(5).cuda().eval()
        qnet=MyQNet_equi('/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/nfmodel/nocs/qnet_equi.yaml').cuda().eval()

        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)

        fea_dict=conv(vertices)
        fea_dict_rot=conv(vertices_rot)
        # out_ori=out_rot[:,:,:,ro_ori]
        pred_scale=torch.ones_like(vertices)[:,0,0]
        out=qnet(vertices,fea_dict,pred_scale)['z_so3']
        out_rot=qnet(vertices_rot.contiguous(),fea_dict_rot,pred_scale)['z_so3']
        out_ori=torch.einsum('ij,bcpi->bcpj',R2,out_rot)

        out_norm=F.normalize(out,dim=-1)
        out_ori_norm=F.normalize(out_ori,dim=-1)
        dot=torch.einsum('bdpi,bdpi->bdp',out_norm,out_ori_norm)
        theta = torch.acos(dot)
        nan_mask = torch.isnan(theta)

        # Replace NaN values with zeros
        theta = torch.where(nan_mask, torch.tensor(0.0).to(theta.device), theta)
        theta=theta.mean()
        print(1)
    def eval_Equi_gcn(argv):
        vertices = torch.randn(4, 1024, 3).cuda()
        fea=torch.rand(vertices.shape[0],1,vertices.shape[1],3).cuda()
        R12=get_anchorsV12()
        R60=get_anchorsV()
        trace_idx_ori, trace_idx_rot = get_relativeV12_index()    # 12(rotation anchors)*12(indices on s2), 12*12
        full_trace_idx_ori,full_trace_idx_rot=get_relativeV_index()


        index=44
        index2=45

        # R=torch.from_numpy(R12[index]).float()
        ro_rot=torch.from_numpy(full_trace_idx_rot[index]).long()
        ro_ori=torch.from_numpy(full_trace_idx_ori[index]).long()

        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()
        ro_rot=torch.from_numpy(full_trace_idx_rot[index]).long().cuda()
        ro_ori=torch.from_numpy(full_trace_idx_ori[index2]).long().cuda()
        conv=Equi_gcn(5).cuda()
        # conv=equi_conv3(dim_in=1,dim_out=32,kernel_size=7,is_surface=True).cuda()

        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)

        out=conv(vertices)
        out_rot=conv(vertices_rot)
        out_ori=out_rot[:,:,:,ro_ori]
        out_diff=out-out_ori
        out_norm=torch.norm(out_diff,dim=-1).sum()



        print(1)
    def eval_neuron(argv):
        vertices = torch.randn(4, 1024, 3).cuda()
        fea=torch.rand(vertices.shape[0],1,vertices.shape[1],3).cuda()
        R12=get_anchorsV12()
        R60=get_anchorsV()
        trace_idx_ori, trace_idx_rot = get_relativeV12_index()    # 12(rotation anchors)*12(indices on s2), 12*12
        full_trace_idx_ori,full_trace_idx_rot=get_relativeV_index()


        index=44
        index2=12

        # R=torch.from_numpy(R12[index]).float()
        # ro_rot=torch.from_numpy(trace_idx_rot[index]).long()
        # ro_ori=torch.from_numpy(trace_idx_ori[index]).long()

        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()
        ro_rot=torch.from_numpy(full_trace_idx_rot[index]).long().cuda()
        ro_ori=torch.from_numpy(full_trace_idx_ori[index]).long().cuda()
        conv=MyNeuron().cuda()
        # conv=equi_conv3(dim_in=1,dim_out=32,kernel_size=7,is_surface=True).cuda()

        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)
        fea_rot=torch.einsum('ij,bcpj->bcpi',R,fea)
        # fea_rot=fea[:,:,:,ro_rot]
        neighbor_index = get_neighbor_index(vertices, 10).cuda()
        out=conv(vertices)[2]
        neighbor_index_rot = get_neighbor_index(vertices_rot, 10).cuda()
        out_rot=conv(vertices_rot)[2]
        # out_ori=out_rot[:,:,:,ro_ori]
        out_ori=torch.einsum('ij,bcpi->bcpj',R,out_rot)

        out_norm=F.normalize(out,dim=-1)
        out_ori_norm=F.normalize(out_ori,dim=-1)
        dot=torch.einsum('bdpi,bdpi->bdp',out_norm,out_ori_norm)
        theta = torch.acos(dot)
        nan_mask = torch.isnan(theta)

        # Replace NaN values with zeros
        theta = torch.where(nan_mask, torch.tensor(0.0).to(theta.device), theta)
        theta=theta.mean()
        print(1)
    def eval_v4(argv):
        vertices = torch.randn(1, 200, 3).cuda()
        R12=get_anchorsV12()
        R60=get_anchorsV()



        index=23
        index2=45



        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()
        conv=Equi_diff_gcn(5).cuda().eval()
        qnet=MyQNet_equi_v4('/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/nfmodel/nocs/qnet_equi_128.yaml').cuda().eval()

        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)

        _,_,fea_dict=conv(vertices)
        _,_,fea_dict_rot=conv(vertices_rot)
        # out_ori=out_rot[:,:,:,ro_ori]
        pred_scale=torch.ones_like(vertices)[:,0,0]
        out=qnet(vertices,fea_dict,pred_scale)['z_so3']
        out_rot=qnet(vertices_rot.contiguous(),fea_dict_rot,pred_scale)['z_so3']
        out_ori=torch.einsum('ij,bcpi->bcpj',R,out_rot)

        out_norm=F.normalize(out,dim=-1)
        out_ori_norm=F.normalize(out_ori,dim=-1)
        dot=torch.einsum('bdpi,bdpi->bdp',out_norm,out_ori_norm)
        theta = torch.acos(dot)
        nan_mask = torch.isnan(theta)

        # Replace NaN values with zeros
        theta = torch.where(nan_mask, torch.tensor(0.0).to(theta.device), theta)
        theta=theta.mean()
        print(1)

    def eval_v5(argv):
        vertices = torch.randn(1, 200, 3).cuda()
        R12=get_anchorsV12()
        R60=get_anchorsV()

        face_to_cube=[(1,4),(2,0),(3,1),(4,2),
                      (0,3),(3,2),(4,3),(0,4),
                      (1,0),(2,1),(4,0),(0,1),
                      (1,2),(2,3),(3,4),(1,3),
                      (0,2),(4,1),(3,0),(2,4)]
        face_to_cube=torch.from_numpy(np.array(face_to_cube))


        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
               (1,2,4),(2,7,8),(3,7,9),(3,5,11),
               (4,5,6),(2,4,10),(2,8,10),(7,8,9),
               (3,9,11),(5,6,11),(4,6,10),(0,8,10),
               (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

        face_normal=vs[faces,:].sum(1)
        face_normal=torch.from_numpy(face_normal).float()
        face_normal=F.normalize(face_normal,dim=-1)

        index=13
        index2=24



        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()
        conv=Equi_diff_gcn_a5(5).cuda().eval()


        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)

        fea=conv(vertices,True).cpu()
        fea_rot=conv(vertices_rot,True).cpu()

        vert_value_1=fea[:,:,:,face_to_cube[:,0]]*(fea[:,:,:,face_to_cube[:,0]]>fea[:,:,:,face_to_cube[:,1]])
        vert_value_1=torch.einsum('b c q n , n i->b c q i',vert_value_1,face_normal)

        vert_value_2=fea_rot[:,:,:,face_to_cube[:,0]]*(fea_rot[:,:,:,face_to_cube[:,0]]>fea_rot[:,:,:,face_to_cube[:,1]])
        vert_value_2=torch.einsum('b c q n , n i->b c q i',vert_value_2,face_normal)

        out_ori=torch.einsum('ij,bcpi->bcpj',R.cpu(),vert_value_2)

        out_diff=vert_value_1-out_ori
        out_norm=torch.norm(out_diff,dim=-1).sum()
        print(1)



    def eval_v6(argv):
        vertices = torch.randn(1, 300, 3).cuda()
        R12=get_anchorsV12()
        R60=get_anchorsV()

        v2colors=[0,0,2,5,3,4,1,1,4,3,5,2]
        color2v=[(0,1),(6,7),(2,11),(4,9),(5,8),(3,10)]
        color_com=[(0,1,2,3,4,5),
                   (0,5,4,3,2,1),
                   (2,5,4,1,0,3),
                   (5,0,1,3,2,4),
                   (3,5,2,0,4,1),
                   (4,0,5,2,1,3),
                   (1,3,4,2,0,5),
                   (1,0,2,4,3,5),
                   (4,1,2,5,0,3),
                   (3,1,4,0,2,5),
                   (5,1,0,4,2,3),
                   (2,4,5,3,0,1)]

        v2colors=torch.from_numpy(np.array(v2colors))
        color2v=torch.from_numpy(np.array(color2v))
        color_com=torch.from_numpy(np.array(color_com))



        roll=np.array([(0,1,2,3,4),(1,2,3,4,0),(2,3,4,0,1),(3,4,0,1,2),(4,0,1,2,3)])
        roll=torch.from_numpy(roll)

        weight=torch.from_numpy(np.array([4e10,4e8,4e6,4e4,4e2]))


        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()


        index=54
        index2=24



        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()
        conv=Equi_diff_gcn_a6(5).cuda().eval()


        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)

        fea=conv(vertices).cpu()
        fea_rot=conv(vertices_rot).cpu()
        vs=vs_

        vert_color_1=fea[:,:,:,color_com][:,:,:,:,0]
        vert_color_1=torch.einsum('bcqv, vi -> bcqvi',vert_color_1,vs)
        vert_color_pair_1=vert_color_1[:,:,:,color2v,:]
        vert_color_roll_1=fea[:,:,:,color_com][:,:,:,:,1:]

        vert_color_roll_1=vert_color_roll_1[:,:,:,:,roll]*weight
        vert_color_roll_1=vert_color_roll_1.sum(-1).max(-1)[0]

        vert_color_pair_index_1=torch.max(vert_color_roll_1[:,:,:,color2v],dim=-1,keepdim=True)[1]
        vert_value_1=torch.gather(vert_color_pair_1,4,vert_color_pair_index_1[:,:,:,:,:,None].repeat(1,1,1,1,1,3)).sum(-2).sum(-2)


        vert_color_2=fea_rot[:,:,:,color_com][:,:,:,:,0]
        vert_color_2=torch.einsum('bcqv, vi -> bcqvi',vert_color_2,vs)
        vert_color_pair_2=vert_color_2[:,:,:,color2v,:]
        vert_color_roll_2=fea_rot[:,:,:,color_com][:,:,:,:,1:]

        vert_color_roll_2=vert_color_roll_2[:,:,:,:,roll]*weight
        vert_color_roll_2=vert_color_roll_2.sum(-1).max(-1)[0]

        vert_color_pair_index_2=torch.max(vert_color_roll_2[:,:,:,color2v],dim=-1,keepdim=True)[1]
        vert_value_2=torch.gather(vert_color_pair_2,4,vert_color_pair_index_2[:,:,:,:,:,None].repeat(1,1,1,1,1,3)).sum(-2).sum(-2)




        out_ori=torch.einsum('ij,bcpi->bcpj',R2.cpu(),vert_value_2)

        out_diff=vert_value_1-out_ori
        out_norm=torch.norm(out_diff,dim=-1).sum()
        print(1)


    def eval_v7(argv):
        vertices = torch.randn(1, 60, 3).cuda()
        R12=get_anchorsV12()
        R60=get_anchorsV()
        face_to_cube=[(1,4),(2,0),(3,1),(4,2),
                      (0,3),(3,2),(4,3),(0,4),
                      (1,0),(2,1),(4,0),(0,1),
                      (1,2),(2,3),(3,4),(1,3),
                      (0,2),(4,1),(3,0),(2,4)]
        face_to_cube=torch.from_numpy(np.array(face_to_cube))


        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
               (1,2,4),(2,7,8),(3,7,9),(3,5,11),
               (4,5,6),(2,4,10),(2,8,10),(7,8,9),
               (3,9,11),(5,6,11),(4,6,10),(0,8,10),
               (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

        face_normal=vs[faces,:].sum(1)
        face_normal=torch.from_numpy(face_normal).float()
        face_normal=F.normalize(face_normal,dim=-1)

        index=13
        index2=24



        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()
        conv=Equi_diff_gcn_a5(5).cuda().eval()
        qnet=MyQNet_equi_an('/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/nfmodel/equi_diff/qnet_equi_128.yaml').cuda().eval()

        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)

        _,_,fea_dict=conv(vertices)
        _,_,fea_dict_rot=conv(vertices_rot)

        pred_scale=torch.ones_like(vertices)[:,0,0]
        out=qnet(vertices,fea_dict,pred_scale)['z_so3']
        out_rot=qnet(vertices_rot.contiguous(),fea_dict_rot,pred_scale)['z_so3']
        out_ori=torch.einsum('ij,bcpi->bcpj',R,out_rot)

        out_norm=F.normalize(out,dim=-1)
        out_ori_norm=F.normalize(out_ori,dim=-1)
        dot=torch.einsum('bdpi,bdpi->bdp',out_norm,out_ori_norm)
        theta = torch.acos(dot)
        nan_mask = torch.isnan(theta)

        # Replace NaN values with zeros
        theta = torch.where(nan_mask, torch.tensor(0.0).to(theta.device), theta)
        theta=theta.mean()
        print(1)

    from absl import app
    # from config.nocs.NFconfig_v8 import *
    from config.equi_diff_nocs.config import *
    # from nfmodel.nocs.NFnetwork_v8 import *
    FLAGS = flags.FLAGS
    app.run(eval_v7)