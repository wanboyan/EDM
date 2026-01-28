import os
import pdb
import random


import torch
print(torch.__version__)
from absl import app
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from config.equi_diff.config import *

from nfmodel.equi_diff.NFnetwork import PIPS_s
from nfmodel.equi_diff.eval_utils import my_eval
from core.models.sim3sdf_vanilla import *
import yaml

FLAGS = flags.FLAGS
from datasets.equi_diff.load_data import PoseDataset
from datasets.equi_diff.load_data_nocs import PoseDataset_nocs
import time

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



    cat_name='combine'


    eval_save=FLAGS.eval_out
    if not os.path.exists(eval_save):
        os.makedirs(eval_save)

    logger = setup_logger('train_log', os.path.join(eval_save, 'eval_result.txt'))
    for key, value in FLAGS.flag_values_dict().items():
        logger.info(key + ':' + str(value))

    # test_dataset = PoseDataset(data_dir=FLAGS.dataset_dir,mode='test')
    test_dataset = PoseDataset_nocs(data_dir=FLAGS.dataset_dir,per_obj='mug',
                                    detection_dir=os.path.join(FLAGS.dataset_dir, 'segmentation_results'))


    network = PIPS_s()
    network = network.to(device)
    if FLAGS.resume_model_name=='latest':
        resume_dir=os.path.join(FLAGS.resume_dir,cat_name)
        check_files=[f for f in os.listdir(resume_dir) if f.startswith('model_')]
        latest_file=max(check_files,key=lambda x: int((x.split('_')[-1]).split('.')[0]))
        resume_path=os.path.join(resume_dir,latest_file)
        print('load',resume_path)
    else:
        resume_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
    # try:
    state_dict=torch.load(resume_path)
    partail_keys=[k for k in state_dict.keys() if 'backbone1'in k or 'diffusion_model' in k]
    partail_dicts={k:state_dict[k] for k in partail_keys}
    network.load_state_dict(state_dict,strict=True)

    cfg_path='./config/equi_diff/vae.yaml'
    decoder_resume_path='./weights/vae_offset/epoch=49499.ckpt'
    with open(cfg_path,'r') as file:
        cfg=yaml.safe_load(file)
    vae=VaeModel(cfg).to(device)
    ckpt = torch.load(decoder_resume_path)
    new_state_dict = {}
    for k,v in ckpt['state_dict'].items():
        new_key = k.replace("network.", "",1)
        new_state_dict[new_key] = v
    vae.load_state_dict(new_state_dict)



    my_eval(network,vae,test_dataset)


if __name__ == "__main__":

    app.run(eval)
