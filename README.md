# EDM



# install
```bash
pip install -r requirements.txt
```

# dataset



# Training

Training is divided into two stages: the first stage trains the point cloud backbone, and the second stage trains the diffusion model. Training is performed on the synthetic dataset [nocs_pack](https://github.com/hughw19/NOCS_CVPR2019), and testing is performed on the real dataset [nocs_real](https://github.com/hughw19/NOCS_CVPR2019).

## Stage 1: Point Cloud Backbone Training

```bash
python engine/train_equi_diff.py \
  --model_save xxx \    # Stage 1 output directory
  --stage 1 \
  --dataset_dir xxx     # Synthetic dataset directory (nocs_pack)
```

## Stage 2: Diffusion Model Training

```bash
python engine/train_equi_diff.py \
  --model_save xxx \    # Stage 2 output directory
  --stage_1_dir xxx \   # Stage 1 save directory
  --stage 2 \
  --dataset_dir xxx     # Synthetic dataset directory (nocs_pack)
```

# Testing

```bash
python -m evaluation/eval_equi_diff.py \
  --resume_dir xxx \    # Stage 2 save directory
  --dataset_dir xxx     # Real dataset directory (nocs_real)
```

# Citation

If you use this code in your research, please cite:

```
B. Wan, Y. Shi, X. Chen and K. Xu, "Equivariant Diffusion Model With A5-Group Neurons for Joint Pose Estimation and Shape Reconstruction," 
in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 6, pp. 4343-4357, June 2025, 
doi: 10.1109/TPAMI.2025.3540593.
```

**BibTeX:**
```bibtex
@article{wan2025equivariant,
  title={Equivariant Diffusion Model With A5-Group Neurons for Joint Pose Estimation and Shape Reconstruction},
  author={Wan, B. and Shi, Y. and Chen, X. and Xu, K.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={47},
  number={6},
  pages={4343--4357},
  year={2025},
  month={June},
  doi={10.1109/TPAMI.2025.3540593}
}
```

**Keywords:** Shape, Diffusion models, Pose estimation, Neurons, Feature extraction, Three-dimensional displays, Image reconstruction, Vectors, Point cloud compression, Noise, Diffusion model, Equivariant network, Pose estimation, Shape reconstruction 