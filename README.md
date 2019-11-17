# Global Voxel Transformer Networks: A Deep Learning Tool for Augmented Microscopy

This is the implementation of the GVTNets proposed in our paper ***Global Voxel Transformer Networks: 
A Deep Learning Tool for Augmented Microscopy*** with Python v3.6 and TensorFlow v1.10 or higher.

## System Requirement

Though this tool can be used with only CPU, we highly recommend to use on GPU(s) with 10GB memory or more. 
Also, it is recommended to use the tool on Linux. Other operating systems are not fully tested.

## Installation

We highly recommend the users to [install Anaconda](https://www.anaconda.com/distribution/) for a simple 
environment setup and installation.

To setup the environment and install the tool, run the following command in the shell.
```
git clone https://github.com/zhengyang-wang/Image2Image.git
cd Image2Image
conda env create -f gvtnet.yml
```

## Usage

At each time, to begin to use the tool, run the following command to activate the virtual environment.
```
conda activate gvtnet
```
After using the tool, you can run the following command to exit the virtual environment.
```
conda deactivate
```

### To train and inference/test on the *Label-free* Datasets and the *CARE* Datasets

We prepared the scripts for these datasets.

### To train and inference/test with your own datasets.

- Prepare your training dataset: (randomly) crop the training image pairs into two sets of patches and save 
them into npz file(s). The npz file(s) can be either A single npz file containing all training data structured as:

			
      {'X': (n_sample, n_channel, (depth,) height, width),
       'Y': (n_sample, n_channel, (depth,) height, width)}
      
    or Multiple npz files where each one contains one training sample structured as:
      
      {'X': (n_channel, (depth,) height, width),
       'Y': (n_channel, (depth,) height, width)}
       
- Training with the dataset: 

      python train.py [--args]
      
     You will need to specify the arguments such as npz_dataset_dir, gpu_id. You can refer to the scripts for the 
     example arguments settings. You can also tune the model parameters by modifying *network_configure.py*.
     
- Predict and evaluate the dataset: the prediction and evaluate accept the tif/tiff files as inputs.

      python predict.py [--args]
      
     and then
     
      python evaluate.py [--args]
      
     You will need to specify the arguments for prediction and evaluation repectivelu such as tiff_dataset_dir, 
     gpu_id. You can refer to the scripts for the example arguments settings.
     
    

