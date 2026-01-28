from __future__ import print_function

import absl.flags as flags

# eval



# train
flags.DEFINE_integer('regular_grid_spilt',9, '')
flags.DEFINE_integer('gt_regular_grid_spilt',19, '')
flags.DEFINE_integer('gt_topk',500, '')

flags.DEFINE_integer('train_pips_t',0, '')


flags.DEFINE_string('generated_gt_dir', '/data_nvme/pips/nocs','')


flags.DEFINE_string('resume_pips_t_dir', '/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/engine/output/v7_pips_t','')
flags.DEFINE_string('resume_pips_t_model_name', 'model_34.pth','')


