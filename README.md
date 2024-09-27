
# Neural-based Estimation of Divergences and Integral Probability Metrics (NEDIPM)

## Overview
This repository contains implementations for neural-based estimation of divergences and integral probability metrics (IPM). The implemented methods cover a wide range of divergences, including various f-divergences, integral probability metrics, and mutual information estimators.

## Class Hierarchy
The structure of the divergence classes is as follows:
```
Divergence
    ├── KLD_DV
    ├── IPM
    │   ├── Wasserstein_GP
    │   └── Wasserstein_GP2
    ├── f_Divergence (LT-based)
    │   ├── KLD_LT
    │   ├── Pearson_chi_squared_LT
    │   ├── squared_Hellinger_LT
    │   ├── Jensen_Shannon_LT
    │   └── alpha_Divergence_LT
    ├── Pearson_chi_squared_HCR
    └── Renyi_Divergence
        ├── Renyi_Divergence_DV
        ├── Renyi_Divergence_CC
            ├── Renyi_Divergence_CC_rescaled
            └── Renyi_Divergence_CC_WCR
```

## Requirements
Everything was tested on cuda 12.5 and cudnn 8.9.2
All dependencies are listed in the `requirements.txt` file. To set up the environment, run the following commands:
```bash
# Create a new conda environment
conda create --name <env> python=3.10.4
pip install -r requirements.txt
```

Additionally, you can install packages individually:
```bash
# Install necessary packages
pip install tensorflow==2.8.0  # For Divergences.py (version 2.8.0 does not support CUDA)
pip install tensorflow_addons==0.16.1  # For the 1d_Gaussian_demo.py
pip install pandas==1.4.2
pip install argparse==1.1
pip install protobuf==3.20.3
pip install matplotlib==3.7.1
pip install scipy
pip install torchsummary

# Install PyTorch with CUDA support if available
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # if CUDA is available
pip3 install torch torchvision torchaudio  # if CUDA is not available

# Install JAX with CUDA support (if needed)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax
```

## Examples
The following are some example use cases available in this repository:
1. **Multivariate Gaussians**: Vary dimension and correlation coefficient (`rho`)
2. **Heavy-tailed distribution**: Varying alpha
3. **Subpopulation detection**: Both synthetic (GMM) and real datasets
4. **Equivariant dataset**: To test on structured data
5. **Image-based tasks**: Divergence estimation with possible CNN-based models
6. **Generation/GANS**: Generating MNIST and CIFAR-10 images

Key parameters to test include sample size, batch size, Lipschitz constant, alpha, and Gamma function space, among others.
All the examples were tested on one GPU, a 4070 Super with 16GB.
## Python Files
- `Divergences.py`: Contains implementations of all the basic divergence families (the test function/discriminator is an input argument).
- One file for each demonstration example (e.g., `1D Gaussian`, `Mixture of Gaussians`, `Subpopulation detection`, etc.).

Additionally, gamma function spaces implemented include continuous & bounded, L-Lipschitz, equivariant, and user-defined.

## How to Run
Here are some example commands to run the scripts:
```bash
# Run an N-dimensional Gaussian example with dimension 1
python N_dim_Gaussian_demo.py --sample_size 10000 --batch_size 1000 --epochs 200 --method KLD-DV --use_GP True --dimension 1

# Run an MNIST GAN example
python mnist_gan.py --method KLD-DV --use_GP True --conditional True

# Run an CIFAR-10 GAN example
python cifar10_gan.py --method KLD-DV --use_GP True --conditional True

# Run an biological hypothesis test example
python Divergence_bio_hypothesis_test_demo.py --p 0.01 --method KLD-DV



## License
Please refer to the LICENSE file for more information.
