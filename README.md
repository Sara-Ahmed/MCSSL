# MC-SSL: Towards Multi-Concept Self-Supervised Learning

This repository contains the official PyTorch self-supervised pretraining, finetuning, and evaluation codes for 
[MCSSL](https://arxiv.org/abs/2111.15340): Towards Multi-Concept Self-Supervised Learning.

# Main Architecture

![](imgs/MCSSL.png)

# Self-supervised Clustering

![](imgs/MCSSL__Cluster.png)


# Self-supervised pre-training
> python -m torch.distributed.launch --nproc_per_node=4 --use_env main_ASiT.py --batch_size 32 --epochs 100 --data_path 'path/to/audio/files' --data-train 'path/to/json/file'

Self-supervised pre-trained models using ASiT can be downloaded from [here .. coming soon](https://drive.google.com/drive/folders/)

# Data Preparation
We mainly employed AudioSet for ASiT pre-training which contains YouTube videos. Please follow [link](https://research.google.com/audioset/download.html) to download and process AudioSet data.

# Acknowledgement
This repository is built using the SiT and the DINO repository.

# Citation
If you use this code for a paper, please cite:

```
@article{atito2021mc,

  title={MC-SSL0. 0: towards multi-concept self-supervised learning},
  
  author={Atito, Sara and Awais, Muhammad and Farooq, Ammarah and Feng, Zhenhua and Kittler, Josef},
  
  journal={arXiv preprint arXiv:2111.15340},
  
  year={2021}
  
}
```
