import os
import pdb
import random


import torch
print(torch.__version__)
from absl import app
import numpy as np

from config.equi_diff.config import *
from tools.training_utils import build_lr_rate,build_diff_optimizer,build_optimizer
from nfmodel.equi_diff.NFnetwork import PIPS_s


FLAGS = flags.FLAGS
from datasets.equi_diff.load_data import PoseDataset

import time

# from creating log
from tensorboardX import SummaryWriter
#SummaryWriter encapsulates everything

from tools.eval_utils import setup_logger, compute_sRT_errors


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

device = 'cuda'
def train(argv):

    torch.backends.cuda.matmul.allow_tf32=False
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)

    cat_name='combine'
    FLAGS.append_flags_into_file(os.path.join(FLAGS.model_save, 'config.txt'))
    cat_model_save=os.path.join(FLAGS.model_save,cat_name)
    if not os.path.exists(cat_model_save):
        os.makedirs(cat_model_save)
    if FLAGS.use_board:
        from tensorboardX import SummaryWriter
        tb_writter = SummaryWriter(cat_model_save)

    logger = setup_logger('train_log', os.path.join(cat_model_save, 'log.txt'))
    for key, value in FLAGS.flag_values_dict().items():
        logger.info(key + ':' + str(value))

    train_dataset = PoseDataset(data_dir=FLAGS.dataset_dir,mode='train')


    # from diffusers import UNet2DModel



    network = PIPS_s()
    network = network.to(device)


    if FLAGS.stage==2:
        resume_path=FLAGS.stage_1_path
        state_dict=torch.load(resume_path)
        partail_keys=[k for k in state_dict.keys() if 'backbone1'in k]
        partail_dicts={k:state_dict[k] for k in partail_keys}
        network.load_state_dict(partail_dicts,strict=False)




    if FLAGS.resume:
        if FLAGS.resume_model_name=='latest':
            check_files=[f for f in os.listdir(os.path.join(FLAGS.resume_dir,cat_name)) if f.startswith('model_')]
            latest_file=max(check_files,key=lambda x: int((x.split('_')[-1]).split('.')[0]))
            resume_path=os.path.join(FLAGS.resume_dir,cat_name,latest_file)
            print('load',resume_path)
        else:
            resume_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
        state_dict=torch.load(resume_path)
        network.load_state_dict(state_dict)
        s_epoch = FLAGS.resume_point
    else:
        s_epoch = 0

    st_time = time.time()
    train_steps = FLAGS.train_steps
    global_step = train_steps * s_epoch


    # record the number iteration
    train_size = train_steps * FLAGS.batch_size
    indices = []
    page_start = - train_size

    #  build optimizer
    param_list = network.build_params()
    optimizer = build_diff_optimizer(param_list)
    # optimizer = build_optimizer(param_list)
    optimizer.zero_grad()   # first clear the grad
    # scheduler = build_lr_rate(optimizer, total_iters=train_steps * FLAGS.total_epoch // FLAGS.accumulate)

    # for i in range(global_step):
    #     scheduler.step()
    #  training iteration, this code is develop based on object deform net
    for epoch in range(s_epoch, FLAGS.total_epoch):




        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                      ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
        # create optimizer and adjust learning rate accordingly
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            data_list = list(range(train_dataset.length))
            for i in range((train_size - len_last) // train_dataset.length + 1):
                random.shuffle(data_list)
                indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start + train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=FLAGS.num_workers, pin_memory=True)
        network.train()
        batch_start=time.time()
        batch_loss_end=time.time()
        #################################
        for i, data in enumerate(train_dataloader, 1):
            batch_start=time.time()
            do_refine=True
            loss_dict= network(depth=data['roi_depth'].to(device),
                               def_mask=data['roi_mask_deform'].to(device),
                               camK=data['cam_K'].to(device),
                               gt_2D=data['roi_coord_2d'].to(device),
                               gt_R=data['rotation'].to(device),
                               gt_t=data['translation'].to(device),
                               aug_rt_t=data['aug_rt_t'].to(device),
                               aug_rt_r=data['aug_rt_R'].to(device),
                               aug_scale=data['aug_scale'].to(device),
                               model_size=data['scale'].to(device),
                               model_points=data['model_points'].to(device),
                               sphere_points=data['sphere_points'].to(device),
                               latent_code=data['latent_code'].to(device),
                               )

            total_loss = loss_dict['loss']
            if FLAGS.use_board:
                tb_writter.add_scalar('lr',optimizer.param_groups[0]["lr"],global_step)
                tb_writter.add_scalar('train_loss',total_loss,global_step)
                if loss_dict['loss_100_latent']>0:
                    tb_writter.add_scalar('loss_100_latent',loss_dict['loss_100_latent'],global_step)
                if loss_dict['loss_100_trans']>0:
                    tb_writter.add_scalar('loss_100_trans',loss_dict['loss_100_trans'],global_step)
                if loss_dict['loss_100_scale']>0:
                    tb_writter.add_scalar('loss_100_scale',loss_dict['loss_100_scale'],global_step)
                if loss_dict['loss_1000_latent']>0:
                    tb_writter.add_scalar('loss_1000_latent',loss_dict['loss_1000_latent'],global_step)
                if loss_dict['loss_1000_trans']>0:
                    tb_writter.add_scalar('loss_1000_trans',loss_dict['loss_1000_trans'],global_step)
                if loss_dict['loss_1000_scale']>0:
                    tb_writter.add_scalar('loss_1000_scale',loss_dict['loss_1000_scale'],global_step)
                tb_writter.add_scalar('nocs_loss',loss_dict['nocs_loss'],global_step)
                tb_writter.add_scalar('var_loss',loss_dict['var_loss'],global_step)




            # backward
            if global_step % FLAGS.accumulate == 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                # weight1=network.weight_model.weight_model.mlp1.layers[0].weight.data.detach().cpu().numpy()
                optimizer.step()
                # scheduler.step()
                # weight2=network.weight_model.weight_model.mlp1.layers[0].weight.data.detach().cpu().numpy()
                # update=weight2-weight1
                # print(update)
                # for name,params in network.diffusion_model.named_parameters():
                #         print(name,params.grad)
                # if FLAGS.use_board and global_step % 10*FLAGS.log_every == 0:
                #     for name,params in network.weight_model.named_parameters():
                #         # print(name,params.grad)
                #         if params.grad is not None:
                #             tb_writter.add_histogram(name,params.grad.cpu().numpy(),global_step)





                optimizer.zero_grad()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)

            batch_dur=time.time()-batch_loss_end
            batch_loss_end=time.time()
            batch_loss_dur=batch_loss_end-batch_start
            global_step += 1

            if i % FLAGS.log_every == 0:
                logger.info('Batch {0} Loss:{1:f}'.format(i, total_loss))
                logger.info('batch_dur {0:f} , bach_loss_dur {1:f}'.format(batch_dur,batch_loss_dur))

        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))

        # save model
        if (epoch + 1) % FLAGS.save_every == 0 or (epoch + 1) == FLAGS.total_epoch:
            torch.save(network.state_dict(), '{0}/model_{1:02d}.pth'.format(cat_model_save, epoch))
            torch.save(optimizer.state_dict(), '{0}/optimizer_{1:02d}.pth'.format(cat_model_save, epoch))


if __name__ == "__main__":

    app.run(train)
