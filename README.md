# Global Voxel Transformer Networks: A Deep Learning Tool for Augmented Microscopy

## Global Voxel Transformer Networks (GVTNets)

![gvtnets](./doc/GVTNets.jpg)

## Augmented Microscopy Tasks

![tasks](./doc/Tasks.jpg)

## System Requirements

We highly recommend using the Linux operating system. Using an nVIDIA GPU with >=11GB memory is highly recommended, although this tool can be used with CPU only.

We used Ubuntu 16.04.6 LTS (GNU/Linux 4.4.0-141-generic x86_64)) and an nVIDIA GeForce RTX 2080 Ti GPU with nVIDIA driver version 430.50 and CUDA version 10.1.

## Installation

### Environment setup

- We highly recommend installing [Anaconda](https://www.anaconda.com/distribution/) for a simple environment setup and installation.
- Download the code:
```
git clone https://github.com/zhengyang-wang/Image2Image.git
cd Image2Image
```
- Create a virtual environment with required packages:
```
conda env create -f gvtnet.yml
```
- Activate the virtual environment:
```
conda activate gvtnet
```
or
```
source activate gvtnet
```
Choose whichever works for you.

## Usage

### Label-free prediction of 3D fluorescence images from transmitted-light microscopy

#### Preparation

- Download [label-free datasets](https://downloads.allencell.org/publication-data/label-free-prediction/index.html).

- Untar all the datasets under one folder. Suppose the folder is `data/`, it should contain 13 datasets named `beta_actin`, `fibrillarin`, etc. In addition, `data/beta_actin/` should contain images only.

- Give execution permission to scripts:
```
chmod +x ./scripts/label-free/*.sh
```

- Change `RAW_DATASET_DIR` in `train.sh`, `train_dic.sh`, `train_membrane.sh` and `predict.sh`, `predict_dic.sh`, `predict_membrane.sh` to the path to your folder that saves all the untarred datasets.

#### Training

- Modify `network_configure.py` according to your design. For users who are not familiar with deep learning, simply copy the content of `network_configures/gvtnet_label-free.py` to `network_configure.py`.

- Train the GVTNets for datasets except `dic_lamin_b1` and `membrane_caax_63x`:
```
./scripts/label-free/train.sh [dataset] [gpu_id] [model_name]
```

- For `dic_lamin_b1` and `membrane_caax_63x`, use `train_dic.sh` and `train_membrane.sh`, respectively:
```
./scripts/label-free/train_dic.sh [gpu_id] [model_name]
./scripts/label-free/train_membrane.sh [gpu_id] [model_name]
```

---

>**Example:**
>>If you want to train a GVTNet called `your-gvtnet` on `beta_actin` using the GPU #1, run:
>>```
>>./scripts/label-free/train.sh beta_actin 1 your-gvtnet
>>```
>>After training, you will find:
>>- Transformed datasets are saved under `save_dir/label-free/beta_actin/datasets/`. This process will only be performed for the first run.
>>- The content in `network_configure.py` is saved as `network_configures/your-gvtnet.py`.
>>- Model checkpoints are saved under `save_dir/label-free/beta_actin/models/your-gvtnet/`.

>**Note**: Always give a different `model_name` when you use a different `network_configure.py`. This tool will used `model_name` to track different network configures.

---

#### Prediction

- Predict the testing set using saved model checkpoints for datasets except `dic_lamin_b1` and `membrane_caax_63x`:
```
./scripts/label-free/predict.sh [dataset] [gpu_id] [model_name] [checkpoint_num]
```

- For `dic_lamin_b1` and `membrane_caax_63x`, use `predict_dic.sh` and `predict_membrane.sh`, respectively:
```
./scripts/label-free/predict_dic.sh [gpu_id] [model_name] [checkpoint_num]
./scripts/label-free/predict_membrane.sh [gpu_id] [model_name] [checkpoint_num]
```

---

>**Example:**
>>If you have trained a GVTNet called `your-gvtnet` on `beta_actin`, and want to make prediction for the testing set with the saved model checkpoints after training for `75,000` minibatch iterations, run:
>>```
>>./scripts/label-free/predict.sh beta_actin 1 your-gvtnet 75000
>>```
>>After prediction, you will find:
>>- Prediction results are saved under `save_dir/label-free/beta_actin/results/your-gvtnet/checkpoint_75000/`.

>**Note**: If your GPU memory is limited, set `gpu_id` to `-1` for CPU prediction.

---

#### Evaluation

- Evaluate the prediction on the testing set for your saved model checkpoints:
```
./scripts/label-free/evaluate_dir.sh [dataset] [model_name] [checkpoint_num]
```

- Users can use `evaluate_file.sh` to evaluate any single prediction.

---

>**Example:**
>>To evaluate the predictions made in the last example, run:
>>```
>>./scripts/label-free/evaluate_dir.sh beta_actin your-gvtnet 75000
>>```

---

#### Use provided pretrained models for reproduction of results in our paper

- Make predictions for datasets except `dic_lamin_b1` and `membrane_caax_63x`:
```
./scripts/label-free/predict.sh [dataset] [gpu_id] gvtnet_label-free_pretrained pretrained
```

- For `dic_lamin_b1` and `membrane_caax_63x`, use `predict_dic.sh` and `predict_membrane.sh`, respectively:
```
./scripts/label-free/predict_dic.sh [gpu_id] gvtnet_label-free_pretrained pretrained
./scripts/label-free/predict_membrane.sh [gpu_id] gvtnet_label-free_pretrained pretrained
```

- Evaluate the results:
```
./scripts/label-free/evaluate_dir.sh [dataset] gvtnet_label-free_pretrained pretrained
```

You will obtain the exact number reported in the supplementary of our paper.

### Content-aware 3D image denoising and 3D to 2D image projection (CARE)

#### Preparation

- Download `Denoising_Planaria.tar.gz`, `Denoising_Tribolium.tar.gz`, `Projection_Flywing.tar.gz` from [CARE datasets](https://publications.mpi-cbg.de/publications-sites/7207/).

- Extract the datasets. Each should contain a `train_data` folder and a `test_data` folder.

- Give execution permission to scripts:
```
chmod +x ./scripts/care_denoising/*.sh
chmod +x ./scripts/care_projection/*.sh
```

- Change `NPZ_DATASET_DIR` in `train_[Planaria|Tribolium|Flywing].sh` to the path to the corresponding `train_data` folder.

- Change `TEST_DATASET_DIR` in `train_[Planaria|Tribolium|Flywing].sh` to the path to the corresponding `test_data` folder.

#### Training

- Modify `network_configure.py` according to your design. For users who are not familiar with deep learning, simply copy the content of `network_configures/gvtnet_care.py` to `network_configure.py`.

- Train the GVTNets:
```
./scripts/care_denoising/train_[Planaria|Tribolium].sh [gpu_id] [model_name]
./scripts/care_projection/train_Flywing.sh [gpu_id] [model_name]
```

---

>**Note**: Always give a different `model_name` when you use a different `network_configure.py`. This tool will used `model_name` to track different network configures.

---

#### Prediction


- Predict the testing set using saved model checkpoints:
```
./scripts/care_denoising/predict_[Planaria|Tribolium].sh [condition] [gpu_id] [model_name] [checkpoint_num]
./scripts/care_projection/predict_Flywing.sh [condition] [gpu_id] [model_name] [checkpoint_num]
```

---

**Note**: If your GPU memory is limited, set `gpu_id` to `-1` for CPU prediction.

---

#### Evaluation

- Evaluate the prediction on the testing set for your saved model checkpoints:
```
./scripts/care_denoising/evaluate_[Planaria|Tribolium].sh [condition] [model_name] [checkpoint_num]
./scripts/care_projection/evaluate_Flywing.sh [condition] [model_name] [checkpoint_num]
```

#### Use provided pretrained models for reproduction of results in our paper

- Make predictions:
```
./scripts/care_denoising/predict_[Planaria|Tribolium].sh [condition] [gpu_id] gvtnet_care_pretrained pretrained
./scripts/care_projection/predict_Flywing.sh [condition] [gpu_id] gvtnet_care_pretrained pretrained
```

- Evaluate the results:
```
./scripts/care_denoising/evaluate_[Planaria|Tribolium].sh [condition] gvtnet_care_pretrained pretrained
./scripts/care_projection/evaluate_Flywing.sh [condition] gvtnet_care_pretrained pretrained
```

You will obtain the exact number reported in the supplementary of our paper.

### To train and inference/test with your own datasets.

- To prepare your training dataset: (randomly) crop the training image pairs into two sets of patches and save 
them into npz file(s). The npz file(s) can be either a single npz file containing all training data structured as:

            
      {'X': (n_sample, n_channel, (depth,) height, width),
       'Y': (n_sample, n_channel, (depth,) height, width)}
      
    or multiple npz files where each one contains one training sample structured as:
      
      {'X': (n_channel, (depth,) height, width),
       'Y': (n_channel, (depth,) height, width)}
      
    If your data contains uncropped images with different sizes, use the later data structure. Check `./datasets/label-free/generate_npz_or_tiff.py` for an example.
       
- To train with the dataset: 

      python train.py [--args]
      
     You will need to specify the arguments, such as npz_dataset_dir, gpu_id. You can refer to the scripts for the 
     example argument settings. You can also tune the model parameters by modifying *network_configure.py*.
     
     
---

   >**Note**: Always give a different `model_name` when you use a different `network_configure.py`. This tool will used `model_name` to track different network configures.

---

     
     Explaination to some arguments:
     ```
     --already_cropped: include it only when training images are already cropped to patches. If not, 
                        you need to specify the --train_patch_size and the image will be automatically 
                        cropped.
     --proj_model: whether to use ProjectionNet to project 3D images to 2D, only used in 3D-to-2D 
                   transform task, e.g. CARE Flywings projection'.
     --offset: whether to add inputs to the outputs (so that the output is considered as an offset of 
               input image). It is applied in CARE models.
     --probalistic: whether to train with probalistic loss, used in CARE models.
     ```
     
- To prepare the prediction and evaluation data: the prediction and evaluation accept the tif/tiff files as inputs. Each
tif/tiff file contains one image of shape 

      [(depth,) height, width]
           
     The ground truth files used for evaluation should have the same names as their corresponding input files and be 
     stored in a different directory to the result or the input files.
     
- To predict and evaluate the dataset: run the following command,

      python predict.py [--args]
      
     and then
     
      python evaluate.py [--args]
      
     You will need to specify the arguments for prediction and evaluation respectively, such as tiff_dataset_dir, 
     gpu_id. You can refer to the scripts for the example argument settings.
     
     Explaination to some arguments:
     ```
     --cropped_prediction: suggested when having a GPU memory problem. The input images will be processed
                           patch by patch and assembled back together. If included, you also need to
                           specify the --predict_patch_size and --overlap.
     --CARE_normalize: include it when you need to use the percentile normalization used in CARE.
     ```
    

