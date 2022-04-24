MVA Generative Modeling projet
==============================

**Authors**: [Arthur Pignet](https://github.com/arthurPignet), [Jie Wang](https://github.com/jie-wang-e)

This repository holds the code for our project on the paper [Autoregresive diffusion models](). The aim of the project was to understand, reimplement if necessary to perfomr an experiment in low dimensionnal setting, and discuss the paper.  
The provided notebook aims to reproduce our experiment on the binarized MNIST dataset. 
We used the code released with the paper for the training part. However, we modified the code in order to have it working, especially for the dataset management, and the sampling which was broken. We recommend to use the notebook in Google Colab. <a href="https://colab.research.google.com/github/arthurPignet/mva-generative-modeling-project/blob/main/ARDM.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

We also of course changed the config file. Unfortunatly, we were not able to run the training for a long time, as we only have access to the free version of colab. Moreover our colabs were also extensively used in other courses, for practicals, and other projects. We thus trained the model as much as possible before being cut by Colab from GPU access. Moreover we were sometimes disconnected, or ran out of memory. All of this to explain that if you want to re-do our experiment from scratch, you might need to re-lauch the training several times, as we did. We advice to re-load the model last checkpoint. 

Project Organization
------------


    │   .gitignore
    │   LICENSE
    │   README.md           <- The top-level README.
    │   requirements.txt    <- List of third parts libraries used in this project.
    |   ARDM.ipynb          <- Notebook used in Colab to run our experiments. 
    │
    ├───autoregressive_diffusion <- [Original paper code](https://github.com/google-research/google-research), with few modifications. 
    │   ├───experiments
    │   │   ├───audio
    │   │   │   ├───arch
    │   │   │   ├───configs
    │   │   │   ├───datasets
    │   │   │   └───model
    │   │   ├───images
    │   │   │   ├───architectures
    │   │   │   ├───eval_scripts
    │   │   │   └───config_simple.py   <- Our config file.   
    │   │   └───language
    │   │       ├───architectures
    │   │       └───configs
    │   ├───model
    │   │   ├───architecture_components
    │   │   └───autoregressive_diffusion
    │   └───utils
    │       └───tests
    ├───results
    │    ├───bin_mnist <- Results of our first failed experiment (network too big).
    │    └───bin_mnist_2   <- Results of our experiment
    │        ├───loss_plots
    │        └───samples  <- Samples from our trained network. 
    │
    └───_can_be_deleted    <- Trash bin (!! git ignored)


Installation 
---------------

In order to run the code in Colab, you will need to install a few more libraries.

```bash
pip install jax
pip install flax
pip install clu

```

Tested configuration
---------------
This installation was tested in Google Colab on GPU instances (free version). <a href="https://colab.research.google.com/github/arthurPignet/mva-generative-modeling-project/blob/main/ARDM.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Result example
---------------
Images generated after 105 epochs of training on binarized MNIST. 

![](https://github.com/arthurPignet/mva-generative-modeling-project/blob/7d991b06999c6ab58fa88cdba7e32bd918d669d4/results/bin_mnist_2/samples/sample_105_epochs.png)

