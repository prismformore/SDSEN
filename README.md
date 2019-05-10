# SDSEN: Self-Refining Deep Symmetry Enhanced Network for Rain Removal

Rain removal aims to extract and remove rain streaks from images. Although convolutional neural network (CNN) based methods have achieved impressive results in this field, they are not equivariant to object rotation, which decreases their generalization capacity for tilted rain streaks. In order to solve this problem, we propose Deep Symmetry Enhanced Network (DSEN). DSEN extracts rotationally equivariant features of rain streaks, and then generates rain layer for image restoration. Furthermore, an efficient selfrefining mechanism is designed to remove accumulated rain streaks more thoroughly. Experimental study verifies the validity of our method, with self-refining DSEN yielding the state-of-the-art performance on both synthetic and real-world rain image datasets.

## Prerequisite
- GrouPy ([Pytorch Implementation](https://github.com/adambielski/GrouPy))
- Python>=3.6
- Pytorch>=1.0.0
- Opencv>=3.1.0
- tensorboard-pytorch

This project is based on [RESCAN](https://github.com/XiaLiPKU/RESCAN).

## Project Structure
- config: contains all codes
    - cal_ssim.py
    - clean.sh
    - dataset.py
    - main.py
    - model.py
    - settings.py
    - show.py
    - tensorboard.sh
- explore.sh
- logdir: holds patches generated in training process
- models: holds checkpoints
- showdir: holds images predicted by the model


## Default Dataset settings
Rain100H: [http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html][9]<br>
Rain800: [https://drive.google.com/drive/folders/0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s][10]

We concatenate the two images(B and O) together as default inputs. If you want to change this setting, just modify config/dataset.py.
Moreover, there should be three folders 'train', 'val', 'test' in the dataset folder.
After download the datasets, don't forget to transform the format!

## Train, Test and Show
    python main.py -a train
    python main.py -a test
    python show.py

## Scripts
- explore.sh: Show the predicted images in browser
- config/tensorboard.sh: Open the tensorboard server
- config/clean.sh: Clear all the training records in the folder

# Cite
Please kindly consider citing our paper:
>@article{sdsen_2018,
  title={Self-Refining Deep Symmetry Enhanced Network for Rain Removal},
  author={Ye, Hanrong and Li, Xia and Liu, Hong and Shi, Wei and Liu, Mengyuan and Sun, Qianru},
  journal={arXiv preprint arXiv:1811.04761},
  year={2018}
}
