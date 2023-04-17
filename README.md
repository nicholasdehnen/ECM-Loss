# Long-tail Detection with Effective Class-Margins 

## Introduction 

This is an re-implementation of [**Long-tail Detection with Effective Class-Margins**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680684.pdf), done as part of the EECS6322 course at York University. 

Find the original paper and its authors here:
> [**Long-tail Detection with Effective Class-Margins**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680684.pdf)             
> [Jang Hyun Cho](https://janghyuncho.github.io/) and [Philipp Kr&auml;henb&uuml;hl](https://www.philkr.net/)

Find the original implementation here:
> https://github.com/janghyuncho/ECM-Loss


## Installation
### Requirements 
The code in this repository was found to be working with the following libraries and corresponding versions:

- Python 3.8+ (probably, only 3.8 was tested)
- PyTorch 0.13.1
- torchvision 0.14.1
- mmdet 2.25.2
- mmcv 1.7.0

### Setup
To setup the code, please follow the commands below:

~~~
# Clone the repo.
git clone https://github.com/nicholasdehnen/ECM-Loss.git
cd ECM-Loss 

# Create conda env.
conda create --name ecm_loss python=3.8 -y 
conda activate ecm_loss

# Install PyTorch.
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 cudatoolkit==11.7 -c pytorch -c nvidia

# Install mmcv.
pip install -U openmim
mim install mmcv-full==1.7.0

# Install mmdet and dependencies.
pip install -e .
pip install mmdet==2.25.2 

# Install lvis-api. 
pip install lvis

# Downgrade numpy to <1.24.0 due to incompatibility
pip install numpy==1.23.1
~~~

### Dataset 
Please download [LVIS dataset](https://www.lvisdataset.org/dataset), and structure the folders as following. 
~~~
data
  ├── lvis_v1
  |   ├── annotations
  │   │   │   ├── lvis_v1_val.json
  │   │   │   ├── lvis_v1_train.json
  │   ├── train2017
  │   │   ├── 000000004134.png
  │   │   ├── 000000031817.png
  │   │   ├── ......
  │   ├── val2017
  │   ├── test2017
~~~

## Training with ECM-Loss 
The authors original training scripts, as well as the ones added as part of this course project can be found [here](https://github.com/janghyuncho/ECM-Loss/tree/main/sh_files/ecm_loss). 

To train using the author's implementation of the ECM-Loss and a *mask-rcnn* framework with *resnet-50* backbone for 12 epochs, use the following command:
~~~
./sh_files/ecm_loss/r50_1x.sh 
~~~

Alternatively, to train using the re-implementation of the ECM-Loss, use one of the following commands:
~~~
./sh_files/ecm_loss/r50_1x_nick.sh
# or
./sh_files/ecm_loss/r50_1x_nick_author_bg_handling.sh
~~~

Note that both Cho's scripts, as well as the ones added here assume a setup with 8 GPUs available. Furthermore, the \*_nick\*.sh scripts assume that the GPUs available have atleast 12GB of VRAM.

## ECM-Loss implementation

* Cho's version of the ECM-Loss: [effective_class_margin_loss.py](https://github.com/nicholasdehnen/ECM-Loss/blob/0af6ce2ccf54b2feb8f4d430335b754699843cae/mmdet/models/losses/effective_class_margin_loss.py)

* Course project re-implementation: [ecm_loss_nick.py](https://github.com/nicholasdehnen/ECM-Loss/blob/d090ffc044c444c5f5956bcc89d420e8ab0b388d/mmdet/models/losses/ecm_loss_nick.py)

* Course project re-implementation + Cho's bg class handling: [ecm_loss_nick_author_bg_handling.py](https://github.com/nicholasdehnen/ECM-Loss/blob/d090ffc044c444c5f5956bcc89d420e8ab0b388d/mmdet/models/losses/ecm_loss_nick_author_bg_handling.py)


## Citation
If you use use ECM Loss, please cite the authors original paper:

	@inproceedings{cho2022ecm,
  		title={Long-tail Detection with Effective Class-Margins},
  		author={Jang Hyun Cho and Philipp Kr{\"a}henb{\"u}hl},
  		booktitle={European Conference on Computer Vision (ECCV)},
  		year={2022}
	}


## Acknowledgement 
ECM Loss is based on [MMDetection](https://github.com/open-mmlab/mmdetection). 
