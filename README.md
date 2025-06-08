# AJTP
Official PyTorch implementation of "Asynchronous Joint-based Temporal Pooling for Skeleton-based Action Recognition". The paper is accepted to TCSVT ([Paper](https://ieeexplore.ieee.org/abstract/document/10685538))

![learnable_mask1](https://github.com/user-attachments/assets/f2f7292e-b359-4d7c-8568-a47b9801c009)


## Prerequisites
Python >= 3.6

PyTorch >= 1.1.0

PyYAML, tqdm, tensorboardX


A dependency file of our experimental environment is provided to install all dependencies by creating a new Anaconda virtual environment and running pip install -r requirements.txt 



```bash
# Run
 pip install -e torchlight
```

## Data Preparation

 ### Download datasets.
There are three datasets to download.
1. NTU RGB+D 120 Skeleton
2. PKUMMD
3. Kinetic400

  #### NTU RGB+D 120
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   -  ```nturgbd_skeletons_s001_to_s017.zip``` (NTU RGB+D 60)
   -  ```nturgbd_skeletons_s018_to_s032.zip ``` (NTU RGB+D 120)
   -  Extract the above files to ```bash ./data/nturgbd_raw ```

 #### PKUMMD
1. Download the dataset from [here.](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html)
   
 #### Kinetic skeleton 400
1. Download dataset from ST-GCN repo: https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton
2. This might be useful if you want to ```wget``` the dataset from Google Drive

### Data Processing

#### Directory Structure
Put the downloaded data into the following directory structure:
```
- data/

  - ntu120/
      - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
  -pkummd/
    - label/   # all the labels 
    - skeleton/    #all the skeletons
  
  - kinetics/kinetics-skeleton/
        - kinetics_train/
          ...
        - kinetics_val/
          ...
        - kinetics_train_label.json
        - keintics_val_label.json

```

#### Generating Data
* Generate NTU RGB+D 120 dataset:
```
 cd ./data/ntu120
 # Get the skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py

```

* Generate PKUMMD dataset:

```
 cd ./data/pkummd
 # Generate data
 python pku_gendata.py
```
* Generate Kinetic skeleton 400 dataset:
```
cd ./data/kinetic
# Generate data 
python kinetics_gendata.py
# Denoise data
python datagen_for_CTR_denoise.py
```

## Training & Testing
### Training

* Change the config file depending on what you want.

```
# Example: training AJTP learnable on NTU RGB+D 120 cross-subject joint-wise learnable pooling and  with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/ctrgcn --model_args pooling_strategy ='joint' --model_args learnable_pooling=True  --device 0
```

* To train the model on NTU RGB+D 60/120 with bone or motion modalities, setting bone or vel arguments in the config file default.yaml or in the command line.

```
# Example: training AJTP learnable on NTU RGB+D 120 cross-subject under bone joint-wise learnable pooling and  with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --work-dir work_dir/ntu120/csub/ctrgcn --model_args pooling_strategy ='joint' --model_args learnable_pooling=True  --device 0

```
### Testing

* To test the trained models saved in <work_dir>, run the following command:
```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0

```

### Acknowledgements
This repo is based on [CTRGCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [MS-G3D](https://github.com/kenziyuliu/MS-G3D) and [STGCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton).

### Citation

Please cite this work if you find it useful

```
@ARTICLE{10685538,
  author={Gunasekara, Shanaka Ramesh and Li, Wanqing and Yang, Jack and Ogunbona, Philip O.},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Asynchronous Joint-Based Temporal Pooling for Skeleton-Based Action Recognition}, 
  year={2025},
  volume={35},
  number={1},
  pages={357-366},
  keywords={Adaptation models;Topology;Skeleton;Transformers;Aggregates;Feature extraction;Circuits and systems;Skeleton-based action recognition;adaptive temporal pooling;key joint selection},
  doi={10.1109/TCSVT.2024.3465845}}

```
### Contact

For any questions, feel free to contact: srg079@uowmail.edu.au
