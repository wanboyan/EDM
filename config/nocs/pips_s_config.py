from __future__ import print_function

import absl.flags as flags




# train

flags.DEFINE_integer('s_topk',50, '')


flags.DEFINE_float('lr_weight_model', 1,'')



flags.DEFINE_integer('use_stable_train',0, '')





flags.DEFINE_string('resume_gt_dir', '/data_nvme/trained_models/pips_gcn_v6_far_far','')
flags.DEFINE_string('resume_gt_model_name', 'model_64.pth','')


