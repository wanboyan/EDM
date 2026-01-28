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
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from losses.nf_loss import *
from nnutils.torch_util import *
import torch.optim as optim
from nnutils.torch_pso_v4 import *

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

        self.embeddings=VADLogVar(FLAGS.cur_cat_model_num,64)
        self.decoder=Decoder(64+3,3,256)


    def forward(self, depth, obj_id, camK,
                gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None,
                pad_points=None, sdf_points=None,do_aug=False,rgb=None,gt_mask=None,cat_name=None,
                model_idx=None,deform_sdf_points=None,
                do_refine=False):

        # FLAGS.sample_method = 'basic'
        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None


        query_nocs=sdf_points
        query_num=query_nocs.shape[1]
        PC=torch.bmm(model_point*nocs_scale.reshape(-1,1,1),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)
        points_defor = torch.randn(PC.shape).to(PC.device)
        PC = PC + points_defor * FLAGS.aug_pc_r
        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        pc_num=PC.shape[1]
        model_idx=model_idx.long()
        if FLAGS.debug:
            query_nocs_deform= sdf_points
        else:
            query_nocs_deform=deform_sdf_points
        vae_loss={}
        vae_loss['KLD']=0
        vae_loss['CD']=0
        # show_open3d(deform_sdf_points[0].detach().cpu().numpy(),sdf_points[0].detach().cpu().numpy())
        bs = PC.shape[0]
        single_scale=torch.norm(mean_shape+gt_s,dim=-1,keepdim=True)
        recon,point_fea,global_fea,feature_dict,feature_dict_detach= self.backbone(pc_center)

        a=30
        if FLAGS.debug==1:
            delta_t1=torch.tensor([[[0.00,0.05,0.0]]]).cuda()
            delta_s1=torch.tensor([[[1]]]).cuda()
            delta_r1=torch.zeros(bs,3,3).cuda()
            for i in range(bs):
                x=torch.Tensor(1).cuda()
                x[0]=40
                y=torch.Tensor(1).cuda()
                y[0]=0
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



        query_bin_first=self.to_bin(sym[0][0],FLAGS.bin_size//10,query_nocs)
        # show_open3d(query_pad_bin_first[0].detach().cpu().numpy(),query_pad_bin_first[0].detach().cpu().numpy())
        query_bin_second=self.to_bin(sym[0][0],FLAGS.bin_size,query_nocs)

        cur_query=torch.bmm((query_nocs*init_shape),init_R.permute(0,2,1))+init_t.reshape(-1,1,3)
        # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query[0].detach().cpu().numpy(),color_2=query_nocs[0].detach().cpu().numpy()+0.5)


        m=torch.nn.LogSoftmax(dim=-1)
        rvecs=[]
        for i in range(bs):
            rvecs.append(cv2.Rodrigues(init_R[i].cpu().numpy())[0][:,0])
        rvecs=np.stack(rvecs)
        new_Rvec=torch.from_numpy(rvecs).float().cuda().requires_grad_()
        new_t=init_t.clone().requires_grad_()
        new_scale=init_shape.clone().requires_grad_()


        def show_dis(show_nocs_np,stage='first',bin_size=FLAGS.bin_size//10,ratios=ratio_dict[cat_name]):
            ratio_x=ratios[0]
            ratio_y=ratios[1]
            ratio_z=ratios[2]
            if sym[0][0]==1:
                x_bin_resolution=FLAGS.pad_radius/bin_size*ratio_x
                y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
                x_start=0
                y_start=(-FLAGS.pad_radius)*ratio_y
                z_bin_resolution=0
                z_start=0
            else:
                x_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_x
                y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
                z_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_z
                x_start=(-FLAGS.pad_radius)*ratio_x
                y_start=(-FLAGS.pad_radius)*ratio_y
                z_start=(-FLAGS.pad_radius)*ratio_z
            starts=[x_start,y_start,z_start]
            bin_resolutions=[x_bin_resolution,y_bin_resolution,z_bin_resolution]
            show_nocs=torch.from_numpy(show_nocs_np).float().cuda().unsqueeze(0)
            show_points=torch.bmm((show_nocs*single_scale[0:1].reshape(-1,1,3)),gt_R[0:1].permute(0,2,1))+gt_t[0:1].reshape(-1,1,3)-center[0:1].reshape(-1,1,3)
            # show_open3d(pc_center[0].detach().cpu().numpy(),show_points[0].detach().cpu().numpy())
            query_num=show_points.shape[1]
            pred_nocs_bin=self.qnet(show_points,feature_dict,stage  ).reshape(1,query_num,3,bin_size)
            pred_nocs_dis=nn.functional.softmax(pred_nocs_bin,dim=-1).detach().cpu().numpy()
            for index in range(query_num):
                for axis in [0,1]:
                    plt_y=pred_nocs_dis[0,index,axis,:]
                    gt_x=show_nocs[0,index,axis].item()
                    plt_x=np.arange(bin_size)*bin_resolutions[axis]+starts[axis]
                    plt.plot(plt_x,plt_y)
                    if sym[0][0]==1 and axis==0:
                        gt_x=np.linalg.norm([show_nocs_np[index,0],show_nocs_np[index,2]])
                    plt.scatter(gt_x,0,color='r')
                    plt.show()
        #
        # show_nocs_np=np.array([[2,-0.3,0.2],[1.5,0.3,0]])
        # # show_dis(show_nocs_np)
        # show_dis(show_nocs_np,'second',FLAGS.bin_size)

        gt_query_deform=torch.bmm((query_nocs_deform.detach()*single_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3)
        gt_query=torch.bmm((query_nocs*single_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3)

        gt_color=torch.Tensor([[[255,0,0]]]).repeat(bs,query_num,1)

        pc_num=pc_center.shape[1]
        pc_color=torch.Tensor([[[0,0,255]]]).repeat(bs,pc_num,1)
        query_pc=torch.cat([pc_center.detach().cpu(),gt_query_deform.detach().cpu()],dim=1)
        query_pc_color=torch.cat([pc_color.detach().cpu(),gt_color.detach().cpu()],dim=1)
        state={'pc':query_pc,'color':query_pc_color,'std':0}

        # model_num=model_point.shape[1]
        # batch_latent_re=batch_latent.unsqueeze(1).expand(-1,model_num,-1)
        # inp=torch.cat([batch_latent_re, model_point], dim=-1)
        # deform=self.decoder(inp)
        # model_deform=model_point+deform
        # show_open3d(model_deform[0].detach().cpu().numpy(),model_deform[1].detach().cpu().numpy())
        # show_open3d(deform[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
        # show_open3d(gt_query_deform[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
        # show_open3d(gt_query[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())


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

            cur_query_deform=torch.bmm((query_nocs_deform.detach()*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)


            if sym[0][0]==1:
                loss=cal_sym_coord_2(cur_query_deform,query_nocs_deform.detach())
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
            cur_s=new_scale

            batch_size=query_nocs.shape[0]
            query_num=query_nocs.shape[1]


            cur_query_deform=torch.bmm((query_nocs_deform.detach()*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)

            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query_deform[0].detach().cpu().numpy())
            pred_nocs_bin=self.qnet(cur_query_deform,feature_dict,stage  ).reshape(batch_size,query_num,3,bin_size)
            pred_nocs_log_dis=m(pred_nocs_bin)
            query_log_prob=torch.gather(pred_nocs_log_dis,-1,query_bin.unsqueeze(-1)).squeeze(-1)
            if sym[0][0]==1:
                log_prob=-query_log_prob[:,:,:2].sum(-1).mean()
            else:
                log_prob=-query_log_prob.sum(-1).mean()
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
            if FLAGS.use_pso and FLAGS.debug:
                def wrapper(pso_size=30,query_bin=None,stage='first',bin_size=FLAGS.bin_size//10):
                    re_feature_dict={}
                    for k1,v1 in feature_dict.items():
                        re_layer={}
                        for k2,v2 in v1.items():
                            re_data=v2.repeat(pso_size,1,1)
                            re_layer[k2]=re_data
                        re_feature_dict[k1]=re_layer
                    re_query_nocs_deform=query_nocs_deform.repeat(pso_size,1,1)
                    re_query_bin=query_bin.repeat(pso_size,1,1)
                    query_num=query_bin.shape[1]
                    m=torch.nn.Softmax(dim=-1)
                    def eval_fun(cur_R,cur_t,cur_s):
                        cur_query=torch.bmm((re_query_nocs_deform*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
                        # show_open3d(cur_query[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
                        with torch.no_grad():
                            pred_nocs_bin=self.qnet(cur_query,re_feature_dict,stage).reshape(pso_size,query_num,3,bin_size)
                            pred_nocs_dis=m(pred_nocs_bin)
                            query_prob=torch.gather(pred_nocs_dis,-1,re_query_bin.unsqueeze(-1)).squeeze(-1)
                            if sym[0][0]==1:
                                prob=(query_prob[:,:,0]*query_prob[:,:,1]).mean(-1)
                            else:
                                prob=(query_prob[:,:,0]*query_prob[:,:,1]*query_prob[:,:,2]).mean(-1)
                            return prob
                    return eval_fun

                assert pc_center.shape[0]==1
                pso_size=216
                iter_num=20
                # eval_fun=wrapper(pso_size,query_bin_first,'first',FLAGS.bin_size//10)
                eval_fun=wrapper(pso_size,query_bin_second,'second',FLAGS.bin_size)
                pso=PSO(surface_points=pc_center[0].cpu().numpy(),size=pso_size,iter_num=iter_num,
                        init_R=init_R[0].cpu().numpy(),eval_fun=eval_fun)
                pso_vec,pso_t,pso_s,_=pso.update_pso()
                new_Rvec=torch.from_numpy(pso_vec[None,:]).float().cuda().requires_grad_()
                new_t=torch.from_numpy(pso_t[None,None]).float().cuda().requires_grad_()
                new_scale=torch.from_numpy(pso_s[None,None]).float().cuda().requires_grad_()


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
                cur_loss=refine(query_bin_first,'first',FLAGS.bin_size//10)
                cur_loss.backward(retain_graph=True)
                # print(new_Rvec.grad)
                return cur_loss
            def closure2():
                opt2.zero_grad()
                cur_loss=refine( query_bin_second,'second',FLAGS.bin_size)
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
                # print('finish1')
                for i in range(1):
                    closure2()
                    opt2.step()
                    pred_grad_2=torch.cat([new_Rvec.grad.reshape(bs,-1).requires_grad_(),
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

            cur_query_deform=torch.bmm((query_nocs_deform.detach()*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            cur_query=torch.bmm((query_nocs.detach()*cur_s.reshape(-1,1,1)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query_deform[0].detach().cpu().numpy(),color_2=query_nocs[0].detach().cpu().numpy()+0.5)
            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query[0].detach().cpu().numpy(),color_2=query_nocs[0].detach().cpu().numpy()+0.5)
            # show_open3d(pc_center[0].detach().cpu().numpy(),gt_query[0].detach().cpu().numpy(),color_2=query_nocs[0].detach().cpu().numpy()+0.5)


            if FLAGS.refine_loss_grad==1:
                loss_deform=0
                gt_grad=gt_grad/torch.norm(gt_grad,dim=-1,keepdim=True).detach()
                pred_grad_1=pred_grad_1/torch.norm(pred_grad_1,dim=-1,keepdim=True)
                pred_grad_2=pred_grad_2/torch.norm(pred_grad_2,dim=-1,keepdim=True)

                fsnet_loss={
                    'Rot1': 0,
                    'Rot2': 0,
                    'Recon': 0,
                    'Tran': 0,
                    'Size':0,
                }
                loss_coord=self.loss_coord(pred_grad_1,gt_grad)+self.loss_coord(pred_grad_2,gt_grad)
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
        loss_dict['interpo_loss']={'Nocs':0,'coord':loss_coord,'deform':loss_deform}
        loss_dict['fsnet_loss'] = fsnet_loss
        loss_dict['compare2init']=compare2init
        loss_dict['vae']=vae_loss
        loss_dict['state']=state
        return loss_dict




    def to_bin(self,sym,bin_size,query_nocs):

        query_nocs_r=query_nocs.clone()
        if sym==1:
            x_bin_resolution=FLAGS.pad_radius/bin_size
            y_bin_resolution=2*FLAGS.pad_radius/bin_size
            x_start=0
            y_start=(-FLAGS.pad_radius)
            z_start=0
            query_nocs_r[:,:,0]=torch.norm(query_nocs_r[:,:,(0,2)],dim=-1)
            query_nocs_r[:,:,2]=0
            query_nocs_bin=torch.zeros_like(query_nocs_r).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
        else:
            x_bin_resolution=2*FLAGS.pad_radius/bin_size
            y_bin_resolution=2*FLAGS.pad_radius/bin_size
            z_bin_resolution=2*FLAGS.pad_radius/bin_size
            x_start=(-FLAGS.pad_radius)
            y_start=(-FLAGS.pad_radius)
            z_start=(-FLAGS.pad_radius)
            query_nocs_bin=torch.zeros_like(query_nocs_r).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
            query_nocs_bin[:,:,2]=torch.clamp(((query_nocs_r[:,:,2]-z_start)/z_bin_resolution),0,bin_size-1).long()
        return query_nocs_bin


    def branch_data_augment(self, PCs, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                         model_point, nocs_scale, obj_ids):
        # augmentation
        bs = len(PCs)
        samplenum=FLAGS.random_points
        PC_1=torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=gt_R.device)
        PC_2=torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=gt_R.device)
        gt_s_1=torch.zeros_like(gt_s)
        gt_s_2=torch.zeros_like(gt_s)
        for i in range(bs):
            obj_id = int(obj_ids[i])
            prop_rt = torch.rand(1)
            PC_new=PCs[i]
            gt_R_new=gt_R[i]
            gt_t_new=gt_t[i]
            gt_s_new=gt_s[i]
            if prop_rt < FLAGS.aug_rt_pro:
                PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC_new, gt_R_new,
                                                         gt_t_new, aug_rt_t[i, ...], aug_rt_r[i, ...])

            gt_t_new=gt_t_new.view(-1)
            aug_bb_1 = torch.rand(3).cuda()
            aug_bb_1 = aug_bb_1.uniform_(0.8, 1.2)
            aug_bb_2 = torch.rand(3).cuda()
            aug_bb_2 = aug_bb_2.uniform_(0.8, 1.2)

            PC_new_1, gt_s_new_1 = defor_3D_bb(PC_new, gt_R_new,
                                               gt_t_new, gt_s_new + mean_shape[i, ...],
                                               sym=sym[i, ...], aug_bb=aug_bb_1)
            gt_s_new_1=gt_s_new_1-mean_shape[i, ...]

            PC_new_2, gt_s_new_2 = defor_3D_bb(PC_new, gt_R_new,
                                               gt_t_new, gt_s_new + mean_shape[i, ...],
                                               sym=sym[i, ...], aug_bb=aug_bb_2)
            gt_s_new_2=gt_s_new_2-mean_shape[i, ...]



            prop_bc = torch.rand(1)
            if prop_bc < FLAGS.aug_bc_pro and (obj_id == 5 or obj_id == 1):
                PC_new_1, gt_s_new_1 = defor_3D_bc(PC_new_1, gt_R_new, gt_t_new,
                                                   gt_s_new_1 + mean_shape[i, ...],
                                                   model_point[i, ...], nocs_scale[i, ...])
                gt_s_new_1=gt_s_new_1-mean_shape[i, ...]
                PC_new_2, gt_s_new_2 = defor_3D_bc(PC_new_2, gt_R_new, gt_t_new,
                                                   gt_s_new_2 + mean_shape[i, ...],
                                                   model_point[i, ...], nocs_scale[i, ...])
                gt_s_new_2=gt_s_new_2-mean_shape[i, ...]


            prop_pc = torch.rand(1)
            if prop_pc < FLAGS.aug_pc_pro:
                PC_new_1 = defor_3D_pc(PC_new_1, FLAGS.aug_pc_r)
                PC_new_2 = defor_3D_pc(PC_new_2, FLAGS.aug_pc_r)

            if FLAGS.sample_method == 'basic':
                l_all = PC_new_1.shape[0]
                if l_all <= 1.0:
                    return None, None
                if l_all >= samplenum:
                    replace_rnd = False
                else:
                    replace_rnd = True
                choose1 = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
                choose2 = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
            elif FLAGS.sample_method == 'farthest':
                choose1=farthest_sample(PC_new_1,samplenum)
                choose2=farthest_sample(PC_new_2,samplenum)

            PC_1[i, ...] = PC_new_1[choose1]
            PC_2[i, ...] = PC_new_2[choose2]
            gt_s_1[i, ...] = gt_s_new_1
            gt_s_2[i, ...] = gt_s_new_2

            gt_R[i]=gt_R_new
            gt_t[i]=gt_t_new
            #  augmentation finish
        return PC_1,PC_2,gt_s_1,gt_s_2, gt_R, gt_t
















    def data_augment(self, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                         model_point, nocs_scale, obj_ids):
        # augmentation
        bs = PC.shape[0]
        for i in range(bs):
            obj_id = int(obj_ids[i])
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


            prop_bb = torch.rand(1)
            model_point_new=model_point[i,...]
            if prop_bb < FLAGS.aug_bb_pro:
                #  R, t, s, s_x=(0.9, 1.1), s_y=(0.9, 1.1), s_z=(0.9, 1.1), sym=None
                PC_new, gt_s_new,model_point_new = defor_3D_bb(PC[i, ...], gt_R[i, ...],
                                               gt_t[i, ...], gt_s[i, ...] + mean_shape[i, ...],
                                               sym=sym[i, ...], aug_bb=aug_bb[i, ...],
                                                               model_points=model_point[i,...],nocs_scale=nocs_scale[i, ...])
                gt_s_new = gt_s_new - mean_shape[i, ...]
                PC[i, ...] = PC_new
                gt_s[i, ...] = gt_s_new

            if prop_bc < FLAGS.aug_bc_pro and (obj_id == 5 or obj_id == 1):
                PC_new, gt_s_new = defor_3D_bc(PC[i, ...], gt_R[i, ...], gt_t[i, ...],
                                               gt_s[i, ...] + mean_shape[i, ...],
                                               model_point_new, nocs_scale[i, ...],obj_id)

                gt_s_new = gt_s_new - mean_shape[i, ...]
                PC[i, ...] = PC_new
                gt_s[i, ...] = gt_s_new
            #  augmentation finish
        return PC, gt_R, gt_t, gt_s


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