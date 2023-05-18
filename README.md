## CoTr: Efficient 3D Medical Image Segmentation by bridging CNN and Transformer

This is the official pytorch implementation of the CoTr:<br />

**Paper: [CoTr: Efficient 3D Medical Image Segmentation
by bridging CNN and Transformer](https://arxiv.org/pdf/2103.03024.pdf
).** 


## Requirements
CUDA 11.0<br />
Python 3.7<br /> 
Pytorch 1.7<br />
Torchvision 0.8.2<br />

## Usage

### 0. Installation


<!-- 
########################################################## This is commented out. ##################################################

* Install Pytorch1.7, nnUNet and CoTr as below
  
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

```

########################################################## This is commented out. ##################################################
-->
#### Docker

```
docker pull stevezeyuzhang/unimiss:1.7.1
```


```

cd nnUNet
pip install -e .

cd CoTr_package
pip install -e .
```


```
export nnUNet_raw_data_base="/code/CoTr/DATASET/nnUNet_raw" 

export nnUNet_preprocessed="/code/CoTr/DATASET/nnUNet_preprocessed" 

export RESULTS_FOLDER="/code/CoTr/DATASET/nnUNet_trained_models" 

source /root/.bashrc 
```

### Fix

[CoTr_package/CoTr/network_architecture/ResTranUnet.py Line](https://github.com/steve-zeyu-zhang/CoTr/blob/BHSD/CoTr_package/CoTr/network_architecture/ResTranUnet.py#L155-L158)





### 1. Data Preparation
* Download [BCV dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
* Preprocess the BCV dataset according to the uploaded nnUNet package.
* Training and Testing ID are in `data/splits_final.pkl`.

### 2. Training 
cd CoTr_package/CoTr/run

* Run `nohup python run_training.py -gpu='0' -outpath='CoTr' 2>&1 &` for training.

### 3. Testing 
* Run `nohup python run_training.py -gpu='0' -outpath='CoTr' -val --val_folder='validation_output' 2>&1 &` for validation.

### 4. Citation
If this code is helpful for your study, please cite:

```
@article{xie2021cotr,
  title={CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation},
  author={Xie, Yutong and Zhang, Jianpeng and Shen, Chunhua and Xia, Yong},
  booktitle={MICCAI},
  year={2021}
}
  
```

### 5. Acknowledgements
Part of codes are reused from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Thanks to Fabian Isensee for the codes of nnU-Net.

### Contact
Yutong Xie (xuyongxie@mail.nwpu.edu.cn)
