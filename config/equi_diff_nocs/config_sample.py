from __future__ import print_function

import absl.flags as flags
import config.equi_diff_nocs.fix_config
import numpy as np


np.bool=np.bool_
np.float=np.float_
np.int=np.int_
# eval
flags.DEFINE_string('eval_out', 'equi_diff_nocs_eval', '')
flags.DEFINE_integer('verbose',0 ,'')
flags.DEFINE_integer('shuffle',1, '')
flags.DEFINE_integer('use_board',1, '')


flags.DEFINE_string('rotation_path','/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/rotation.pt', '')
flags.DEFINE_string('occ_r_path','/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/occ_r.pt', '')
flags.DEFINE_string('latent_dir','/data_sata/pack/latent/equi_inv_52499','')

flags.DEFINE_string('occ_latent_dir','/data_sata/pack/latent/occ_24999','')


flags.DEFINE_string('stage_1_dir','/data_sata/pack/weight/nocs_stage_1','')
# flags.DEFINE_string('stage_1_dir','/data_sata/pack/weight/nocs_a5_stage_1','')
# flags.DEFINE_string('stage_1_path','/data_sata/pack/weight/nocs_a5_stage_1/combine/model_159.pth','')

flags.DEFINE_integer('query_num', 500, '')

flags.DEFINE_integer('stage',2, '')

flags.DEFINE_integer('use_occ',0, '')

flags.DEFINE_integer('use_fuse',0, '')

flags.DEFINE_integer('use_res',0, '')

flags.DEFINE_integer('use_simple',1, '')

flags.DEFINE_integer('use_score',0, '')
flags.DEFINE_integer('use_ddm',0, '')
flags.DEFINE_integer('use_pointe',0, '')
flags.DEFINE_integer('use_michel',0, '')

flags.DEFINE_integer('use_origin',0, '')








flags.DEFINE_string('fea_type','a12','')
# flags.DEFINE_string('fea_type','a5','')

# flags.DEFINE_integer('use_12_to_5',1, '')

flags.DEFINE_string('per_cat','camera','')









# flags.DEFINE_string('pred','noise','')
flags.DEFINE_string('pred','start','')













# flags.DEFINE_string('qnet_config','qnet_equi_512.yaml', '')
flags.DEFINE_string('qnet_config','qnet_equi_256.yaml', '')
# flags.DEFINE_string('qnet_config','qnet_equi_128.yaml', '')


# flags.DEFINE_list('dim_list',[128,128,256,256,512],'')
flags.DEFINE_list('dim_list',[64,64,128,128,256],'')
# flags.DEFINE_list('dim_list',[32,32,64,64,128],'')

flags.DEFINE_float('lr_nocs', 3.0, '')
flags.DEFINE_float('lr_diff', 1.0, '')








flags.DEFINE_string('dataset_dir', '/data_sata/pack/per_cat_dataset', 'path to the dataset')
flags.DEFINE_string('cloud_dir', '/data_sata/pack/cloud', 'path to the dataset')





# point selection



flags.DEFINE_string('device', 'cuda:0', '')
flags.DEFINE_integer("num_workers", 4, "cpu cores for loading dataset")
flags.DEFINE_integer('batch_size',4, '')
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
flags.DEFINE_string('model_save', 'output/equi_diff_nocs_a5_stage_2', 'path to save checkpoint')


# resume
flags.DEFINE_integer('resume',0, '1 for resume, 0 for training from the start')
flags.DEFINE_string('resume_dir', '/data_sata/pack/weight/nocs_stage_2','')
# flags.DEFINE_string('resume_model_name', 'model_74.pth','')
flags.DEFINE_string('resume_model_name', 'latest','')

flags.DEFINE_integer('resume_point', 60, 'the epoch to continue the training')






