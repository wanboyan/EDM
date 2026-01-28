import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net_v4 import GCN3D_segR,Rot_red,Rot_green,MyQNet,Pose_Ts,Point_center_res_cate,\
    VADLogVar,Decoder
curdir=os.path.dirname(os.path.realpath(__file__))
qnet_config_file=os.path.join(curdir,'qnet.yaml')
from network.point_sample.pc_sample import *
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from losses.nf_loss import *
from nnutils.torch_util import *
import torch.optim as optim
from nnutils.torch_pso import *
from pyTorchChamferDistance.chamfer_distance import ChamferDistance


def KLD(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    KLD = torch.mean(KLD)
    return KLD
class NFPose(nn.Module):
    def __init__(self):
        super(NFPose, self).__init__()
        self.qnet=MyQNet(qnet_config_file)
        self.rot_green = Rot_green(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.rot_red = Rot_red(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.backbone=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        self.embeddings=VADLogVar(FLAGS.cur_cat_model_num,128)
        self.decoder=Decoder(128,FLAGS.keypoint_num*3,512)
        if FLAGS.feat_for_ts:
            self.ts=Pose_Ts(F=FLAGS.feat_c_ts,k=FLAGS.Ts_c)
        else:
            self.ts=Point_center_res_cate()
        self.loss_fs_net = fs_net_loss()
        self.loss_nocs=nocs_loss()
        self.loss_consistency=consistency_loss()
        self.loss_inter=inter_loss()
        self.loss_coord=nn.SmoothL1Loss(beta=0.5,reduction='mean')
        # self.loss_coord=nn.MSELoss(reduction='mean')
        self.loss_coord_sym=nn.SmoothL1Loss(beta=0.5,reduction='none')
        self.loss_bin_fun=nn.CrossEntropyLoss(reduce=False)
        self.loss_chamfer=ChamferDistance()



    def forward(self, depth, obj_id, camK,
                gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None,
                pad_points=None, sdf_points=None,do_aug=False,rgb=None,gt_mask=None,cat_name=None,
                model_idx=None,
                do_refine=False):

        # FLAGS.sample_method = 'basic'
        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None
        query_nocs=sdf_points
        model_idx=model_idx.long()
        batch_latent_dict = self.embeddings(model_idx)
        batch_latent = batch_latent_dict['latent_code']
        batch_mu = batch_latent_dict['mu']
        batch_logvar = batch_latent_dict['logvar']
        kld=KLD(batch_mu, batch_logvar)
        vae_loss={}
        vae_loss['KLD']=kld*FLAGS.kld_w
        keypoint_nocs=self.decoder(batch_latent).reshape(bs,-1,3)
        dist1, dist2 =self.loss_chamfer(keypoint_nocs,query_nocs)
        loss_chamfer = torch.mean(dist1) + torch.mean(dist2)
        vae_loss['CD']=loss_chamfer*FLAGS.cd_w

        PC=torch.bmm(model_point*nocs_scale.reshape(-1,1,1),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)
        points_defor = torch.randn(PC.shape).to(PC.device)
        PC = PC + points_defor * FLAGS.aug_pc_r
        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        single_scale=torch.norm(mean_shape+gt_s,dim=-1,keepdim=True)
        # show_open3d((sdf_points*mean_shape.reshape(-1,1,3))[0].detach().cpu().numpy(),query_nocs[0].detach().cpu().numpy())
        bs = PC.shape[0]
        recon,point_fea,global_fea,feature_dict= self.backbone(pc_center)

        a=30
        if FLAGS.debug==1:
            delta_t1=torch.tensor([[[0.00,0.00,0.0]]]).cuda()
            delta_s1=torch.tensor([[[0.6]]]).cuda()
            delta_r1=torch.zeros(bs,3,3).cuda()
            for i in range(bs):
                x=torch.Tensor(1).cuda()
                x[0]=0
                y=torch.Tensor(1).cuda()
                y[0]=30
                z=torch.Tensor(1).cuda()
                z[0]=0
                delta_r1[i] = get_rotation_torch(x, y, z)
            init_R=torch.bmm(gt_R,delta_r1)
        else:
            delta_t1 = torch.rand(bs, 1, 3).cuda()
            delta_t1 = delta_t1.uniform_(-0.05, 0.05)
            delta_s1 = torch.rand(bs, 1, 1).cuda()
            delta_s1 = delta_s1.uniform_(0.6, 1.4)
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
        init_R_=torch.cat([init_R,init_R[:,:,:1]],dim=-1)
        gt_R_=torch.cat([gt_R,gt_R[:,:,:1]],dim=-1)
        gt_vec=rotation_matrix_to_angle_axis(gt_R_)
        init_vec=rotation_matrix_to_angle_axis(init_R_)
        init_t=gt_t.reshape(-1,1,3)+delta_t1
        init_t=init_t.reshape(-1,3)-center.reshape(-1,3)
        init_shape=single_scale.reshape(-1,1,1)*delta_s1





        cur_keypoint=torch.bmm((keypoint_nocs*init_shape),init_R.permute(0,2,1))+init_t.reshape(-1,1,3)
        # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query[0].detach().cpu().numpy())


        m=torch.nn.LogSoftmax(dim=-1)
        rvecs=[]
        for i in range(bs):
            rvecs.append(cv2.Rodrigues(init_R[i].cpu().numpy())[0][:,0])
        rvecs=np.stack(rvecs)
        new_Rvec=torch.from_numpy(rvecs).float().cuda().requires_grad_()
        new_t=init_t.clone().requires_grad_()
        new_scale=init_shape.clone().requires_grad_()



        gt_keypoint=torch.bmm((keypoint_nocs.detach()*single_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3)
        gt_query=torch.bmm((query_nocs*single_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3)




        def cal_sym_coord_2(cur_query,query_nocs):
            base=6
            sym_Rs=torch.zeros([base,3,3],dtype=torch.float32, device=query_nocs.device)
            for i in range(base):
                theta=float(i/base)*2*math.pi
                sym_Rs[i]=torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                                        [0, 1, 0],
                                        [-math.sin(theta), 0, math.cos(theta)]])
            sym_query_nocs=torch.matmul(query_nocs.unsqueeze(1),sym_Rs.permute(0,2,1).unsqueeze(0))
            sym_gt_query=torch.matmul(sym_query_nocs*single_scale.reshape(-1,1,1,1),gt_R.permute(0,2,1).unsqueeze(1))+gt_t.reshape(-1,1,1,3)-center.reshape(-1,1,1,3)
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
            cur_s=new_scale
            keypoint_nocs_detach=keypoint_nocs.detach()
            cur_keypoint=torch.bmm((keypoint_nocs_detach*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            cur_query=torch.bmm((query_nocs*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query[0].detach().cpu().numpy())

            if sym[0][0]==1:
                loss=cal_sym_coord_2(cur_keypoint,keypoint_nocs_detach)
            else:
                loss=self.loss_coord(cur_keypoint,gt_keypoint)
            # print(loss)
            return loss




        def refine(stage='first'):

            R_list=[]
            for i in range(bs):
                R_list.append(Rodrigues.apply(new_Rvec[i]))
            cur_R=torch.stack(R_list,dim=0)
            cur_t=new_t
            cur_s=new_scale

            batch_size=keypoint_nocs.shape[0]
            keypoint_num=keypoint_nocs.shape[1]


            cur_keypoint=torch.bmm((keypoint_nocs*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            cur_query=torch.bmm((query_nocs*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)

            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_keypoint[0].detach().cpu().numpy(),color_2=query_nocs[0].detach().cpu().numpy()+0.5)

            pred_nocs_bin=self.qnet(cur_keypoint,feature_dict,stage).reshape(batch_size,keypoint_num,keypoint_num)
            pred_nocs_log_dis=m(pred_nocs_bin)
            index=torch.arange(keypoint_num).reshape(1,keypoint_num,1).repeat(batch_size,1,1).cuda()
            query_log_prob=torch.gather(pred_nocs_log_dis,-1,index).squeeze(-1)
            if sym[0][0]==1:
                raise
            else:
                log_prob=-query_log_prob.mean()
            return log_prob


        track_grads=[]
        track_Rvecs=[]
        track_ts=[]
        track_scales=[]
        track_steps=10
        pred_grad_1=None
        pred_grad_2=None
        if do_refine:
            opt0=torch.optim.Adam([new_Rvec,new_t,new_scale], lr=0.02)
            def gen_track():
                for i in range(track_steps):
                    opt0.zero_grad()
                    gt_loss=gt_refine()
                    gt_loss.backward()
                    track_Rvecs.append(new_Rvec.clone().detach().cpu())
                    track_ts.append(new_t.clone().detach().cpu())
                    track_scales.append(new_scale.clone().detach().cpu())
                    track_grads.append(torch.cat([new_Rvec.grad.clone().reshape(bs,-1),
                                              new_t.grad.clone().reshape(bs,-1),
                                              new_scale.grad.clone().reshape(bs,-1)],dim=-1))
                    opt0.step()
            if FLAGS.debug==0:
                gen_track()
                track_Rvecs=torch.stack(track_Rvecs,dim=0).transpose(0,1)
                track_ts=torch.stack(track_ts,dim=0).transpose(0,1)
                track_scales=torch.stack(track_scales,dim=0).transpose(0,1)
                track_grads=torch.stack(track_grads,dim=0).transpose(0,1)
                track_choose=torch.randint(low=0,high=track_steps,size=[bs])
                batch_choose=torch.arange(bs)

                new_Rvec=track_Rvecs[batch_choose,track_choose].cuda().requires_grad_()
                new_t=track_ts[batch_choose,track_choose].cuda().requires_grad_()
                new_scale=track_scales[batch_choose,track_choose].cuda().requires_grad_()
                gt_grad=track_grads[batch_choose,track_choose].cuda().requires_grad_()


            if FLAGS.train_optimizer=='adam':
                # if FLAGS.debug==1:
                opt1=torch.optim.Adam([new_Rvec,new_t,new_scale], lr=0.01)
                opt2=torch.optim.Adam([new_Rvec,new_t,new_scale], lr=0.01)
                # #
                # else:
                # opt1=torch.optim.SGD([new_Rvec,new_t,new_scale], lr=0.01)
                # opt2=torch.optim.SGD([new_Rvec,new_t,new_scale], lr=0.01)
            else:
                opt1=optim.LBFGS([new_Rvec,new_t,new_scale],lr=1,max_iter=20,line_search_fn="strong_wolfe")
                opt2=optim.LBFGS([new_Rvec,new_t,new_scale],lr=1,max_iter=20,line_search_fn="strong_wolfe")




            def closure1():
                opt1.zero_grad()
                cur_loss=refine()
                cur_loss.backward(retain_graph=True)
                # print(new_Rvec.grad)
                return cur_loss
            def closure2():
                opt2.zero_grad()
                cur_loss=refine()
                cur_loss.backward(retain_graph=True)
                # print(new_Rvec.grad)
                return cur_loss
            if FLAGS.train_optimizer=='adam':
                for i in range(1):
                    closure1()
                    opt1.step()
                    pred_grad_1=torch.cat([new_Rvec.grad.reshape(bs,-1).requires_grad_(),
                                       new_t.grad.reshape(bs,-1).requires_grad_(),
                                       new_scale.grad.reshape(bs,-1).requires_grad_()],dim=-1)

            else:
                opt1.step(closure1)
                print('finish1')
                opt1.step(closure2)
                print('finish2')
            R_list=[]
            for i in range(bs):
                R_list.append(Rodrigues.apply(new_Rvec[i]))
            cur_R=torch.stack(R_list,dim=0)
            cur_t=new_t
            cur_s=new_scale

            cur_query=torch.bmm((query_nocs*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query[0].detach().cpu().numpy(),color_2=query_nocs[0].detach().cpu().numpy()+0.5)
            # show_open3d(pc_center[0].detach().cpu().numpy(),gt_query[0].detach().cpu().numpy(),color_2=query_nocs[0].detach().cpu().numpy()+0.5)


            if FLAGS.refine_loss_grad==1:
                gt_grad=gt_grad/torch.norm(gt_grad,dim=-1,keepdim=True).detach()
                pred_grad_1=pred_grad_1/torch.norm(pred_grad_1,dim=-1,keepdim=True)
                fsnet_loss={
                    'Rot1': 0,
                    'Rot2': 0,
                    'Recon': 0,
                    'Tran': 0,
                    'Size':0,
                }
                loss_coord=self.loss_coord(pred_grad_1,gt_grad)
                compare2init={
                    'Rot1': 0,
                    'Rot2': 0,
                    'Recon': 0,
                    'Tran': 0,
                    'Size':0,
                }

        loss_dict={}
        loss_dict['consistency_loss']={'consistency':0}
        loss_dict['inter_loss'] = {
            'inter_r':0,
            'inter_t':0,
            'inter_nocs':0,
        }
        loss_dict['interpo_loss']={'Nocs':0,'coord':loss_coord}
        loss_dict['fsnet_loss'] = fsnet_loss
        loss_dict['compare2init']=compare2init
        loss_dict['vae']=vae_loss
        return loss_dict





    def to_bin(self,ratios,sym,bin_size,query_nocs):
        ratio_x=ratios[0]
        ratio_y=ratios[1]
        ratio_z=ratios[2]
        query_nocs_r=query_nocs.clone()
        if sym==1:
            x_bin_resolution=FLAGS.pad_radius/bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
            x_start=0
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_bin_resolution=0
            z_start=0
            query_nocs_r[:,:,0]=torch.norm(query_nocs_r[:,:,(0,2)],dim=-1)
            query_nocs_r[:,:,2]=0
            query_nocs_bin=torch.zeros_like(query_nocs_r).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
        else:
            x_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
            z_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_z
            x_start=(-FLAGS.pad_radius)*ratio_x
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_start=(-FLAGS.pad_radius)*ratio_z
            query_nocs_bin=torch.zeros_like(query_nocs_r).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
            query_nocs_bin[:,:,2]=torch.clamp(((query_nocs_r[:,:,2]-z_start)/z_bin_resolution),0,bin_size-1).long()
        return query_nocs_bin

    def one_branch(self,PC,sdf_points,pad_points,mean_shape,gt_s,gt_R,gt_t,sym,cat_name,ori_gt_s=None):
        real_scale=mean_shape+gt_s
        if ori_gt_s is not None:
            ori_real_scale=mean_shape+ori_gt_s
            max_ori_real_scale=torch.max(ori_real_scale,dim=-1)[0]
            pad_points=pad_points*(max_ori_real_scale.reshape(-1,1,1))/(ori_real_scale.reshape(-1,1,3))
        else:
            max_real_scale=torch.max(real_scale,dim=-1)[0]
            pad_points=pad_points*(max_real_scale.reshape(-1,1,1))/(real_scale.reshape(-1,1,3))
        query_nocs=torch.cat([sdf_points,pad_points],dim=1)
        query_points=torch.bmm((query_nocs*real_scale.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)
        # show_open3d(PC[0].cpu().detach().numpy(),query_points[0].cpu().detach().numpy())
        # show_open3d(PC[0].cpu().detach().numpy(),PC[0].cpu().detach().numpy())
        pc_num=PC.shape[1]
        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        recon,point_fea,global_fea,feature_dict= self.backbone(pc_center)
        # point_fea=torch.cat([point_fea,global_fea.unsqueeze(1).repeat(1,pc_num,1)],dim=-1)
        p_green_R=self.rot_green(point_fea.permute(0,2,1))
        p_red_R=self.rot_red(point_fea.permute(0,2,1))
        if FLAGS.feat_for_ts:
            feat_for_ts = torch.cat([point_fea, pc_center], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        else:
            feat_for_ts = pc_center
            objs=torch.zeros_like(feat_for_ts[:,0,0])
            T, s = self.ts(feat_for_ts.permute(0, 2, 1),objs)
        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)

        Pred_T = T + center.squeeze(1)  # bs x 3
        Pred_s = s  # this s is
        recon=recon+center

        pred_fsnet_list = {
            'Rot1': p_green_R,
            'Rot2': p_red_R,
            'Recon': recon,
            'Tran': Pred_T,
            'Size': Pred_s,
        }

        query_num=query_points.shape[1]
        batch_size=query_points.shape[0]
        query_points_center=query_points-center
        pred_nocs_bin=self.qnet(query_points_center,feature_dict).reshape(batch_size,query_num,3,FLAGS.bin_size).permute(0,-1,1,2).contiguous()
        class_mean_shape=mean_shape[0]
        max_class_mean_shape=class_mean_shape.max()
        ratio_x=ratio_dict[cat_name][0]
        ratio_y=ratio_dict[cat_name][1]
        ratio_z=ratio_dict[cat_name][2]

        if sym[0][0]==1:
            x_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_y
            x_start=0
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_bin_resolution=0
            z_start=0
            query_nocs[:,:,0]=torch.norm(query_nocs[:,:,(0,2)],dim=-1)
            query_nocs[:,:,2]=0
            query_nocs_bin=torch.zeros_like(query_nocs).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
        else:
            x_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_y
            z_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_z
            x_start=(-FLAGS.pad_radius)*ratio_x
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_start=(-FLAGS.pad_radius)*ratio_z
            query_nocs_bin=torch.zeros_like(query_nocs).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
            query_nocs_bin[:,:,2]=torch.clamp(((query_nocs[:,:,2]-z_start)/z_bin_resolution),0,FLAGS.bin_size-1).long()

        bin_list={
            'query_nocs_bin':query_nocs_bin,
            'query_points':query_points,
            'pred_nocs_bin':pred_nocs_bin,
            'x_bin_resolution':x_bin_resolution,
            'y_bin_resolution':y_bin_resolution,
            'z_bin_resolution':z_bin_resolution,
            'x_start':x_start,
            'y_start':y_start,
            'z_start':z_start,
        }
        return pred_fsnet_list,bin_list




    def build_params(self,):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []

        # pose
        params_lr_list.append(
            {
                "params": self.backbone.parameters(),
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
        if FLAGS.use_prior:
            params_lr_list.append(
                {
                    "params": self.qnet.qnet.parameters(),
                    "lr": float(FLAGS.lr) * FLAGS.lr_interpo,
                    "betas":(0.9, 0.99)
                }
            )
            params_lr_list.append(
                {
                    "params": self.qnet.fc_out.parameters(),
                    "lr": 0,
                }
            )
        else:
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
        params_lr_list.append(
            {
                "params": self.embeddings.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_embedding,

            }
        )
        params_lr_list.append(
            {
                "params": self.decoder.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_decoder,

            }
        )
        return params_lr_list

    def load_prior(self,cat_name,prior_dir,prior_name):
        #'qnet.fc_out.weight', 'qnet.fc_out.bias'
        cat_model_path=os.path.join(prior_dir,cat_name,prior_name)
        state_dict=torch.load(cat_model_path)
        own_dict=self.state_dict()
        for name in ['qnet.fc_out.weight', 'qnet.fc_out.bias']:
            own_dict[name].copy_(state_dict[name])
        print('load prior')