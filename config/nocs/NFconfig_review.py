from __future__ import print_function

import absl.flags as flags
import config.nocs.pips_s_config
import config.nocs.fix_config
import numpy as np
np.bool=np.bool_
np.float=np.float_
np.int=np.int_
# eval
flags.DEFINE_string('eval_out', 'evaluation/review', '')

flags.DEFINE_float('stage',2,'')

flags.DEFINE_string('pic_save_dir', 'pips_eval_all_cat', '')
flags.DEFINE_integer('verbose',0 ,'')
flags.DEFINE_integer('debug',0 ,'')
flags.DEFINE_integer('shuffle',0, '')
flags.DEFINE_integer('use_board',1, '')
flags.DEFINE_integer('use_pick', 0, '')
flags.DEFINE_float('pick_ratio', 0.0, '')

flags.DEFINE_integer('nu_nocs',0 ,'')

flags.DEFINE_integer('softmax',1 ,'')
flags.DEFINE_integer('use_pool',0 ,'')
flags.DEFINE_string('tovec_version','v3' ,'')
flags.DEFINE_float('atten_factor',3,'')
flags.DEFINE_integer('use_global', 0, '')


flags.DEFINE_string('per_obj', '', '')
# train
flags.DEFINE_string('rotation_path','/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/rotation.pt', '')
flags.DEFINE_string('debug_path','/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/debug.pt', '')


flags.DEFINE_float('rotation_noise_angle', 20, '')



flags.DEFINE_integer('scale_invariant',0, '')

flags.DEFINE_string('backbone','gcn_equi3', '')
# flags.DEFINE_string('backbone','gcn', '')


flags.DEFINE_string('qnet_version','v5','')


# flags.DEFINE_string('qnet_config','qnet_equi_128.yaml', '')
flags.DEFINE_string('qnet_config','qnet_equi_512.yaml', '')
# flags.DEFINE_string('qnet_config','qnet.yaml', '')


flags.DEFINE_list('dim_list',[128,128,256,256,512],'')
# flags.DEFINE_list('dim_list',[32,32,64,64,128],'')
flags.DEFINE_integer('equi_neighbor_num',10, '')
#
flags.DEFINE_integer('fix_kernel',0, '')

flags.DEFINE_integer('use_pred_scale', 0, '')






# datasets
flags.DEFINE_integer('obj_c', 6, 'nnumber of categories')
flags.DEFINE_string('eval_dataset', 'Real', 'CAMERA or CAMERA+Real')


flags.DEFINE_string('dataset', 'Real', 'CAMERA or CAMERA+Real')


flags.DEFINE_string('dataset_dir', '/data_nvme/nocs_data', 'path to the dataset')
flags.DEFINE_string('detection_dir', '/data_nvme/nocs_data/segmentation_results', 'path to detection results')
flags.DEFINE_string('dataset_size','small','big for whole dataset')
flags.DEFINE_integer('big_in_whole',1,'')




# point selection
flags.DEFINE_integer('random_points', 1028, 'number of points selected randomly')
flags.DEFINE_string('sample_method', 'farthest', 'basic')
# flags.DEFINE_string('sample_method', 'basic', 'basic')
flags.DEFINE_string('down_sample_method', 'farthest', 'basic')
# flags.DEFINE_string('down_sample_method', 'basic', 'basic')

# train parameters
# train##################################################
flags.DEFINE_integer("train", 1, "1 for train mode")
# flags.DEFINE_integer('eval', 0, '1 for eval mode')
flags.DEFINE_string('device', 'cuda:0', '')
# flags.DEFINE_string("train_gpu", '0', "gpu no. for training")
flags.DEFINE_integer("num_workers", 4, "cpu cores for loading dataset")
flags.DEFINE_integer('batch_size',4, '')
flags.DEFINE_integer('total_epoch', 70, 'total epoches in training')
flags.DEFINE_integer('train_steps', 2000, 'number of batches in each epoch')  # batchsize is 8, then 3000
#####################space is not enough, trade time for space####################
flags.DEFINE_integer('accumulate', 1, '')   # the real batch size is batchsize x accumulate

# test parameters

# for different losses




# training parameters
# learning rate scheduler
flags.DEFINE_float('lr', 3e-4, '')
 # initial learning rate w.r.t basic lr
flags.DEFINE_float('lr_pose', 1.0, '')
flags.DEFINE_integer('lr_decay_iters', 50, '')  # some parameter for the scheduler### optimizer  ####
flags.DEFINE_string('lr_scheduler_name', 'flat_and_anneal', 'linear/warm_flat_anneal/')
flags.DEFINE_string('anneal_method', 'cosine', '')

flags.DEFINE_float('anneal_point', 0.5, '')

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
flags.DEFINE_string('model_save', 'output/pips_equi5', 'path to save checkpoint')


# resume
flags.DEFINE_integer('resume',0, '1 for resume, 0 for training from the start')
flags.DEFINE_string('resume_dir', '/data_nvme/trained_models/pips_equi3_wo','')
flags.DEFINE_string('resume_model_name', 'model_69.pth','')

flags.DEFINE_integer('resume_point', 39, 'the epoch to continue the training')






