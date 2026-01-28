import os
import pdb
import random


import torch
print(torch.__version__)
from absl import app
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from config.equi_diff_nocs.config_occ import *

from nfmodel.equi_diff_nocs.NFnetwork import PIPS_s
from nfmodel.equi_diff_nocs.eval_utils import my_eval_occ
from core.models.sim3sdf_vanilla import *
from equi_diff_models.my_combined_model import CombinedModel
import yaml

FLAGS = flags.FLAGS
from datasets.equi_diff_nocs.load_data import PoseDataset
from datasets.equi_diff_nocs.load_data_nocs import PoseDataset_nocs
import time
import json

# from creating log
from tensorboardX import SummaryWriter
#SummaryWriter encapsulates everything

from tools.eval_utils import setup_logger, compute_sRT_errors
from nfmodel.lmo.eval_utils import *

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
def eval(argv):

    torch.backends.cuda.matmul.allow_tf32=False

    network_cats=['container','mug']

    cat_name=FLAGS.per_cat


    eval_save=FLAGS.eval_out
    if not os.path.exists(eval_save):
        os.makedirs(eval_save)

    logger = setup_logger('train_log', os.path.join(eval_save, 'eval_result.txt'))
    for key, value in FLAGS.flag_values_dict().items():
        logger.info(key + ':' + str(value))

    # test_dataset = PoseDataset(data_dir=FLAGS.dataset_dir,mode='train',per_cat=cat_name)
    test_dataset = PoseDataset_nocs(data_dir=FLAGS.dataset_dir,per_obj=cat_name,
                                    detection_dir=os.path.join(FLAGS.dataset_dir, 'segmentation_results'))


    network_dict={}
    for network_cat in network_cats:
        network = PIPS_s()
        network = network.to(device)
        if FLAGS.resume_model_name=='latest':
            resume_dir=os.path.join(FLAGS.resume_dir,network_cat)
            check_files=[f for f in os.listdir(resume_dir) if f.startswith('model_')]
            latest_file=max(check_files,key=lambda x: int((x.split('_')[-1]).split('.')[0]))
            resume_path=os.path.join(resume_dir,latest_file)
            print('load',resume_path)
        else:
            resume_path=os.path.join(FLAGS.resume_dir,network_cat,FLAGS.resume_model_name)

        state_dict=torch.load(resume_path)
        network.load_state_dict(state_dict,strict=True)
        network.eval()
        network_dict[network_cat]=network

    cfg_path='/home/wanboyan/Documents/cmr-master/extern/Diffusion-SDF-main/config/stage1_sdf/specs.json'
    resume_path='/data_sata/pack/weight/vae_nocs_occ/epoch=24999.ckpt'
    result_dir='./stage2_occ_nocs'
    specs = json.load(open(cfg_path))
    vae=CombinedModel.load_from_checkpoint(resume_path,specs=specs).to(device)
    vae.eval()


    my_eval_occ(network_dict,vae,test_dataset,result_dir)


if __name__ == "__main__":

    app.run(eval)
