from __future__ import print_function

import absl.flags as flags
import config.equi_diff.fix_config
import numpy as np


np.bool=np.bool_
np.float=np.float_
np.int=np.int_
# eval
flags.DEFINE_string('eval_out', 'equi_diff_eval', '')
flags.DEFINE_integer('verbose',0 ,'')
flags.DEFINE_integer('shuffle',1, '')
flags.DEFINE_integer('use_board',1, '')


flags.DEFINE_string('decoder_version', 'v1', '')
flags.DEFINE_integer('use_vae', 1, '')

flags.DEFINE_string('rotation_path','./weights/rotation.pt', '')
flags.DEFINE_string('latent_dir','./weights/latent/latent_49499','')


flags.DEFINE_string('stage_1_path','/data_sata/pack/stage_1/model_189.pth','')

flags.DEFINE_integer('stage',1, '')

flags.DEFINE_integer('mid_con',1, '')
flags.DEFINE_integer('ddm',0, '')
flags.DEFINE_integer('use_fuse',0, '')
flags.DEFINE_integer('use_simple',0, '')
flags.DEFINE_integer('use_occ',0, '')
flags.DEFINE_integer('use_var',0, '')

flags.DEFINE_integer('fix_attention',0, '')
flags.DEFINE_integer('a_sim',0, '')
flags.DEFINE_integer('regular_transformer',0, '')



flags.DEFINE_string('tmp','v1', '')
flags.DEFINE_string('fea_type','a12', '')

# flags.DEFINE_string('pred','noise','')
flags.DEFINE_string('pred','start','')












flags.DEFINE_string('qnet_config','qnet_equi_128.yaml', '')

# flags.DEFINE_list('dim_list',[128,128,256,256,512],'')
flags.DEFINE_list('dim_list',[32,32,64,64,128],'')

flags.DEFINE_float('lr_nocs', 3.0, '')








# flags.DEFINE_string('dataset_dir', '/data_nvme/nocs_data', 'path to the dataset')
flags.DEFINE_string('dataset_dir', '/data_sata/pack/nocs_pack', 'path to the dataset')

# flags.DEFINE_string('detection_dir', '/data_nvme/nocs_data/segmentation_results', 'path to the detection results')




# point selection



flags.DEFINE_string('device', 'cuda:0', '')
flags.DEFINE_integer("num_workers", 4, "cpu cores for loading dataset")
flags.DEFINE_integer('batch_size',8, '')
flags.DEFINE_integer('total_epoch', 200, 'total epoches in training')
flags.DEFINE_integer('train_steps', 2000, 'number of batches in each epoch')  # batchsize is 8, then 3000
#####################space is not enough, trade time for space####################
flags.DEFINE_integer('accumulate', 1, '')   # the real batch size is batchsize x accumulate




flags.DEFINE_float('lr', 1e-4, '')
flags.DEFINE_integer('lr_decay_iters', 50, '')  # some parameter for the scheduler### optimizer  ####
flags.DEFINE_string('lr_scheduler_name', 'flat_and_anneal', 'linear/warm_flat_anneal/')
flags.DEFINE_string('anneal_method', 'cosine', '')
flags.DEFINE_float('anneal_point', 0.9, '')
flags.DEFINE_string('optimizer_type', 'Ranger', '')
flags.DEFINE_float('weight_decay', 0.0, '')
flags.DEFINE_float('warmup_factor', 0.001, '')
flags.DEFINE_integer('warmup_iters', 1000, '')
flags.DEFINE_string('warmup_method', 'linear', '')
flags.DEFINE_float('gamma', 0.1, '')
flags.DEFINE_float('poly_power', 0.9, '')

# save parameters
#model to save
flags.DEFINE_integer('save_every', 5, '')  # save models every 'save_every' epoch
flags.DEFINE_integer('log_every', 10, '')  # save log file every 100 iterations
flags.DEFINE_string('model_save', 'output/equi_diff_mid_con', 'path to save checkpoint')


# resume
flags.DEFINE_integer('resume',0, '1 for resume, 0 for training from the start')
flags.DEFINE_string('resume_dir', '/data_sata/pack/weight/stage_2_mid_con_v1','')
flags.DEFINE_string('resume_model_name', 'model_314.pth','')


flags.DEFINE_integer('resume_point', 60, 'the epoch to continue the training')






