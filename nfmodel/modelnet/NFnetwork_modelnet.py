import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net_v4 import GCN3D_segR,Rot_red,Rot_green,MyQNet,Pose_Ts,Point_center_res_cate, \
    VADLogVar,Decoder
curdir=os.path.dirname(os.path.realpath(__file__))
qnet_config_file=os.path.join(curdir,'qnet.yaml')
from network.point_sample.pc_sample import *
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v_modelnet
from losses.fs_net_loss import fs_net_loss
from losses.nf_loss import *
from nnutils.torch_util import *
import torch.optim as optim
from nnutils.torch_pso import *



class NFPose(nn.Module):
    def __init__(self):
        super(NFPose, self).__init__()
        self.qnet=MyQNet(qnet_config_file)
        self.rot_green = Rot_green(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.rot_red = Rot_red(F=FLAGS.feat_c_R,k=FLAGS.R_c)

        self.backbone1=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        if FLAGS.two_back:
            self.backbone2=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        if FLAGS.feat_for_ts:
            self.ts=Pose_Ts(F=FLAGS.feat_c_ts,k=FLAGS.Ts_c)
        else:
            self.ts=Point_center_res_cate()
        self.loss_fs_net = fs_net_loss()
        self.loss_consistency=consistency_loss()
        self.loss_inter=inter_loss()
        self.loss_coord=nn.SmoothL1Loss(beta=0.5,reduction='mean')
        self.loss_coord_sym=nn.SmoothL1Loss(beta=0.5,reduction='none')
        self.loss_bin_fun=nn.CrossEntropyLoss(reduce=False)



    def forward_fsnet(self,point_fea,pc_center,center):
        feat_for_ts = pc_center
        objs=torch.zeros_like(feat_for_ts[:,0,0])
        T, s = self.ts(feat_for_ts.permute(0, 2, 1),objs)
        p_green_R=self.rot_green(point_fea.permute(0,2,1))
        p_red_R=self.rot_red(point_fea.permute(0,2,1))
        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)
        pred_fsnet_list = {
            'Rot1': p_green_R,
            'Rot2': p_red_R,
            'Tran': T + center.squeeze(1),
            'Size': s,
        }
        return pred_fsnet_list

    def forward(self,cloud, gt_R, gt_t,  sym=None,
                aug_rt_t=None, aug_rt_r=None,  model_point=None,
                pad_points=None, sdf_points=None,cat_name=None):
        bs=cloud.shape[0]
        gt_t=gt_t[:,0,:]
        PC=cloud
        PC, gt_R, gt_t= self.data_augment(PC, gt_R, gt_t,aug_rt_t,aug_rt_r)

        # transformed_model_point=torch.bmm(model_point,gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)
        # show_open3d(PC[0].detach().cpu().numpy(),transformed_model_point[0].detach().cpu().numpy())


        pad_nocs=pad_points
        pad_num=pad_nocs.shape[0]

        query_nocs=model_point
        query_num=query_nocs.shape[1]

        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        pc_num=PC.shape[1]
        vae_loss={}
        vae_loss['KLD']=0
        vae_loss['CD']=0
        bs = PC.shape[0]

        recon,point_fea,global_fea,feature_dict,feature_dict_detach= self.backbone1(pc_center)


        if FLAGS.two_back:
            recon,point_fea,global_fea,_,_= self.backbone2(pc_center)

        a=10
        delta_t1 = torch.rand(bs, 1, 3).cuda()
        delta_t1 = delta_t1.uniform_(-0.05, 0.05)
        delta_s1 = torch.rand(bs, 1, 3).cuda()
        delta_r1=torch.zeros(bs,3,3).cuda()
        for i in range(bs):
            x=torch.Tensor(1).cuda()
            x.uniform_(-a,a)
            y=torch.Tensor(1).cuda()
            y.uniform_(-a,a)
            z=torch.Tensor(1).cuda()
            z.uniform_(-a,a)
            delta_r1[i] = get_rotation_torch(x, y, z)
        init_R=torch.bmm(delta_r1,gt_R)
        init_t=gt_t.reshape(-1,1,3)+delta_t1
        init_t=init_t.reshape(-1,3)-center.reshape(-1,3)


        m=torch.nn.LogSoftmax(dim=-1)
        rvecs=[]
        for i in range(bs):
            rvecs.append(cv2.Rodrigues(init_R[i].cpu().numpy())[0][:,0])
        rvecs=np.stack(rvecs)
        new_Rvec=torch.from_numpy(rvecs).float().cuda().requires_grad_()
        new_t=init_t.clone().requires_grad_()

        query_nocs_deform=query_nocs
        pad_nocs_deform=pad_nocs


        gt_pad_camera=torch.bmm((pad_nocs.detach()),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()
        gt_query_deform=torch.bmm((query_nocs_deform.detach()),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3)

        gt_query_camera=torch.bmm((query_nocs),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3)
        cur_query_camera=torch.bmm((query_nocs),init_R.permute(0,2,1))+init_t.reshape(-1,1,3)
        # show_open3d(gt_pad_camera[4].detach().cpu().numpy(),pc_center[4].detach().cpu().numpy())

        pad_bin_first_deform,_=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size//10,pad_nocs_deform)
        pad_bin_second_deform,_=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size,pad_nocs_deform)
        if FLAGS.min_loss:
            base=6
            pad_num=pad_nocs.shape[1]
            batch_size=pad_nocs.shape[0]
            sym_Rs=torch.zeros([base,3,3],dtype=torch.float32, device=pad_nocs.device)
            for i in range(base):
                theta=float(i/base)*2*math.pi
                sym_Rs[i]=torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                                        [0, 1, 0],
                                        [-math.sin(theta), 0, math.cos(theta)]])
            sym_pad_nocs_deform=torch.matmul(pad_nocs_deform.unsqueeze(1),sym_Rs.permute(0,2,1).unsqueeze(0))

            sym_pad_bin_second_deform,_=self.to_bin(cat_name,0,FLAGS.bin_size,sym_pad_nocs_deform.reshape(-1,pad_num,3))
            sym_pad_bin_second_deform=sym_pad_bin_second_deform.reshape(batch_size,base,pad_num,3)

        query_bin_second_deform,_=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size,query_nocs_deform)


        if FLAGS.use_nocs_loss:
            pred_pad_bin_dict=self.qnet(gt_pad_camera,feature_dict)



        def do_nocs_loss(stage,pad_bin,bin_size):
            batch_size=pad_bin.shape[0]
            pad_num=pad_bin.shape[1]
            pred_bin=pred_pad_bin_dict[stage].reshape(batch_size,pad_num,3,bin_size).permute(0,-1,1,2).contiguous()
            if sym[0][0]==0:
                return self.loss_bin_fun(pred_bin,pad_bin).mean()*FLAGS.interpo_w
            else:
                return self.loss_bin_fun(pred_bin[:,:,:,(0,2)],pad_bin[:,:,(0,2)]).mean()*FLAGS.interpo_w



        if FLAGS.use_nocs_loss:
            nocs_loss2=do_nocs_loss('second',pad_bin_second_deform,FLAGS.bin_size)
            nocs_loss=nocs_loss2
        else:
            nocs_loss=0
        def cal_sym_coord(cur_query,query_nocs):
            base=20
            sym_Rs=torch.zeros([base,3,3],dtype=torch.float32, device=query_nocs.device)
            for i in range(base):
                theta=float(i/base)*2*math.pi
                sym_Rs[i]=torch.tensor([[math.cos(theta),  math.sin(theta),0],
                                        [-math.sin(theta), math.cos(theta),0,],
                                        [0,0,1]])
            sym_query_nocs=torch.matmul(query_nocs.unsqueeze(1),sym_Rs.permute(0,2,1).unsqueeze(0))
            sym_gt_query=torch.matmul(sym_query_nocs,gt_R.permute(0,2,1).unsqueeze(1))+gt_t.reshape(-1,1,1,3)-center.reshape(-1,1,1,3)
            loss_coord_sym=self.loss_coord_sym(cur_query.unsqueeze(1),sym_gt_query)
            loss_coord_sym=loss_coord_sym.mean(-1).mean(-1)
            loss_coord_sym=loss_coord_sym.min(dim=1)[0].mean()
            return loss_coord_sym

        def gt_refine():

            R_list=[]
            for i in range(bs):
                R_list.append(Rodrigues.apply(new_Rvec[i]))
            cur_R=torch.stack(R_list,dim=0)
            cur_t=new_t
            cur_query_deform=torch.bmm((query_nocs_deform.detach()),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query_deform[0].detach().cpu().numpy())
            if sym[0][0]==1:
                loss=cal_sym_coord(cur_query_deform,query_nocs_deform.detach())
            else:
                loss=self.loss_coord(cur_query_deform,gt_query_deform)
            # print(loss)
            return loss




        def refine(query_bin,stage='first',bin_size=FLAGS.bin_size//10):

            R_list=[]
            for i in range(bs):
                R_list.append(Rodrigues.apply(new_Rvec[i]))
            cur_R=torch.stack(R_list,dim=0)
            cur_t=new_t
            batch_size=query_nocs.shape[0]




            cur_query_deform=torch.bmm((query_nocs_deform.detach()),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query_deform[0].detach().cpu().numpy())
            pred_nocs_bin=self.qnet(cur_query_deform,feature_dict)[stage].reshape(batch_size,query_num,3,bin_size)

            pred_nocs_log_dis=m(pred_nocs_bin)
            query_log_prob=torch.gather(pred_nocs_log_dis,-1,query_bin.unsqueeze(-1)).squeeze(-1)
            if sym[0][0]==1:
                log_prob=-query_log_prob[:,:,(0,2)].sum(-1).mean()
            else:
                log_prob=-query_log_prob.sum(-1).mean()
            return log_prob


        track_grads=[]
        track_Rvecs=[]
        track_ts=[]
        track_scales=[]
        track_steps=15
        pred_grad_1=None
        pred_grad_2=None

        opt0=torch.optim.Adam([new_Rvec,new_t], lr=0.01)
        def gen_track():
            for i in range(track_steps):
                opt0.zero_grad()
                gt_loss=gt_refine()
                gt_loss.backward()
                track_Rvecs.append(new_Rvec.clone().detach().cpu())
                track_ts.append(new_t.clone().detach().cpu())
                track_grads.append(torch.cat([new_Rvec.grad.clone().reshape(bs,-1),
                                          new_t.grad.clone().reshape(bs,-1)],dim=-1))
                opt0.step()



        if FLAGS.use_refine_loss:

            gen_track()
            track_Rvecs=torch.stack(track_Rvecs,dim=0).transpose(0,1)
            track_ts=torch.stack(track_ts,dim=0).transpose(0,1)
            track_grads=torch.stack(track_grads,dim=0).transpose(0,1)
            track_choose=torch.randint(low=0,high=track_steps,size=[bs])
            batch_choose=torch.arange(bs)

            new_Rvec=track_Rvecs[batch_choose,track_choose].cuda().requires_grad_()
            new_t=track_ts[batch_choose,track_choose].cuda().requires_grad_()
            gt_grad=track_grads[batch_choose,track_choose].cuda().requires_grad_()
            # show_open3d(gt_query_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())



            opt2=torch.optim.SGD([new_Rvec,new_t], lr=0.01)






            def closure2():
                opt2.zero_grad()
                cur_loss=refine( query_bin_second_deform,'second',FLAGS.bin_size)
                cur_loss.backward()
                # print(new_Rvec.grad)
                return cur_loss


                # print('finish1')
            for i in range(1):
                closure2()
                opt2.step()
                pred_grad_2=torch.cat([new_Rvec.grad.reshape(bs,-1).requires_grad_(),
                                       new_t.grad.reshape(bs,-1).requires_grad_()],dim=-1)


            gt_grad=gt_grad/torch.norm(gt_grad,dim=-1,keepdim=True).detach()
            # pred_grad_1=pred_grad_1/torch.norm(pred_grad_1,dim=-1,keepdim=True)
            pred_grad_2=pred_grad_2/torch.norm(pred_grad_2,dim=-1,keepdim=True)
            # loss_grad=(self.loss_coord(pred_grad_1,gt_grad)+self.loss_coord(pred_grad_2,gt_grad))*FLAGS.grad_w
            loss_grad=self.loss_coord(pred_grad_2,gt_grad)*FLAGS.grad_w
        else:
            loss_grad=0




        compare2init={
            'Rot1': 0,
            'Rot2': 0,
            'Recon': 0,
            'Tran': 0,
            'Size':0,
        }
        if FLAGS.use_fsnet:
            name_fs_list=['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran',]
            pred_fsnet_list=self.forward_fsnet(point_fea,pc_center,center)
            gt_green_v, gt_red_v = get_gt_v_modelnet(gt_R)
            gt_fsnet_list = {
                'Rot1': gt_green_v,
                'Rot2': gt_red_v,
                'Tran': gt_t,
            }
            fsnet_loss=self.loss_fs_net(name_fs_list,pred_fsnet_list,gt_fsnet_list,sym)
        else:
            fsnet_loss={
                'Rot1': 0,
                'Rot2': 0,
                'Rot1_cos':0,
                'Rot2_cos':0,
                'Rot_r_a':0,
                'Tran': 0,
                'Size': 0
            }
        loss_dict={}
        loss_dict['consistency_loss']={'consistency':0}
        loss_dict['inter_loss'] = {
            'inter_r':0,
            'inter_t':0,
            'inter_nocs':0,
        }
        loss_dict['interpo_loss']={'Nocs':nocs_loss,'grad':loss_grad,'deform':0}
        loss_dict['fsnet_loss'] = fsnet_loss
        loss_dict['compare2init']=compare2init
        loss_dict['vae']=vae_loss
        return loss_dict




    def to_bin(self,cat_name,sym,bin_size,pad_nocs):
        pad_nocs_r=pad_nocs.clone()
        if sym==1:
            x_bin_resolution=FLAGS.pad_radius/bin_size
            y_bin_resolution=0
            z_bin_resolution=2*FLAGS.pad_radius/bin_size
            x_start=0
            y_start=0
            z_start=(-FLAGS.pad_radius)
            pad_nocs_r[:,:,0]=torch.norm(pad_nocs_r[:,:,(0,1)],dim=-1)
            pad_nocs_r[:,:,1]=0
            pad_nocs_bin=torch.zeros_like(pad_nocs_r).long()
            pad_nocs_bin[:,:,0]=torch.clamp(((pad_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,2]=torch.clamp(((pad_nocs_r[:,:,2]-z_start)/z_bin_resolution),0,bin_size-1).long()
        else:
            x_bin_resolution=2*FLAGS.pad_radius/bin_size
            y_bin_resolution=2*FLAGS.pad_radius/bin_size
            z_bin_resolution=2*FLAGS.pad_radius/bin_size
            x_start=(-FLAGS.pad_radius)
            y_start=(-FLAGS.pad_radius)
            z_start=(-FLAGS.pad_radius)
            pad_nocs_bin=torch.zeros_like(pad_nocs_r).long()
            pad_nocs_bin[:,:,0]=torch.clamp(((pad_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,1]=torch.clamp(((pad_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,2]=torch.clamp(((pad_nocs_r[:,:,2]-z_start)/z_bin_resolution),0,bin_size-1).long()
        pad_bin_value = torch.zeros((3,bin_size)).to(pad_nocs_r.device)
        pad_bin_value[0]=x_start+torch.arange(bin_size)*x_bin_resolution
        pad_bin_value[1]=y_start+torch.arange(bin_size)*y_bin_resolution
        pad_bin_value[2]=z_start+torch.arange(bin_size)*z_bin_resolution
        return pad_nocs_bin,pad_bin_value


    def data_augment(self, PC, gt_R, gt_t, aug_rt_t, aug_rt_r,):
        # augmentation
        bs = PC.shape[0]

        for i in range(bs):
            prop_rt = torch.rand(1)
            if prop_rt < FLAGS.aug_rt_pro:
                PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC[i, ...], gt_R[i, ...],
                                                         gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
                PC[i, ...] = PC_new
                gt_R[i, ...] = gt_R_new
                gt_t[i, ...] = gt_t_new.view(-1)

            prop_bc = torch.rand(1)
            # only do bc for mug and bowl


            prop_pc = torch.rand(1)
            if prop_pc < FLAGS.aug_pc_pro:
                PC_new = defor_3D_pc(PC[i, ...], FLAGS.aug_pc_r)
                PC[i, ...] = PC_new



        return PC, gt_R, gt_t,


    def build_params(self,):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []

        # pose
        params_lr_list.append(
            {
                "params": self.backbone1.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_backbone,
            }
        )
        if FLAGS.two_back:
            params_lr_list.append(
                {
                    "params": self.backbone2.parameters(),
                    "lr": float(FLAGS.lr) * FLAGS.lr_backbone,
                }
            )
        params_lr_list.append(
            {
                "params": self.rot_red.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_rot,
            }
        )
        params_lr_list.append(
            {
                "params": self.rot_green.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_rot,
            }
        )

        params_lr_list.append(
            {
                "params": self.qnet.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_interpo,
                "betas":(0.9, 0.99)
            }
        )
        params_lr_list.append(
            {
                "params": self.ts.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_ts,

            }
        )



        return params_lr_list

