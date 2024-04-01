TITLE: Neural-based Estimation of Divergences and Integral Probability Metrics (NEDIPM)



CLASS HIERARCHY
---------------
Divergence
    KLD_DV
    IPM
	Wasserstein_GP
	Wasserstein_GP2
    f_Divergence (LT-based)
	KLD_LT
	Pearson_chi_squared_LT
	squared_Hellinger_LT
	Jensen_Shannon_LT
    	alpha_Divergence_LT
    alpha_Divergence_?? (Scaling optimised)
    chi_squared_?? (Optimized)
    Renyi_Divergence
	Renyi_Divergence_DV
	Renyi_Divergence_CC
	    Renyi_Divergence_CC_rescaled
	    Renyi_Divergence_CC_WCR




REQUIREMENTS
------------
All the requirements are in the requirements txt file.
To create a new enviornment with the specific requirements just run the commands: 
$ conda create --name <env> python=3.10.4
$ pip install -r requirements.txt

conda create --name <env> python=3.10.4
pip install tensorflow==2.8.0 (for the Divergences.py) (version 2.8.0 does not support CUDA)
pip install tensorflow_addons==0.16.1 (for the 1d_Gaussian_demo.py)
pip install pandas==1.4.2
pip install argparse==1.1
pip install protobuf==3.20.3
pip install matplotlib==3.7.1
pip install scipy
pip install torchsummary

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (if cuda is available)
pip3 install torch torchvision torchaudio (if cuda is not available)

(pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html)
pip install flax



LIST OF EXAMPLES
----------------
a. 1D Gaussians: fixed or varying mu & sigma
b. Multivariate Gaussians: vary dimension and correlation coefficient rho
c. Heavy-tailed distribution and varying alpha
d. Subpopulation detection: both synthetic (GMM) and real datasets
e. An equivariant dataset
f. Define 1-2 tasks for images that require divergence estimation (CNN-based?)

- How detailed shall we be?
There are several quantities to test:
sample size, batch size, Lipschitz constant, alpha, Gamma function space, ...



PYTHON FILES
------------
- Divergences.py contains all the basic families of divergences (the test function/discriminator is an input argument)

- One file per demonstration example (1D Gaussian, Mixture of Gaussians/heavy-tailed, subpopulation detection, ...)

- Implement also GANs? (Not sure if we want to...)


- MI.py contains all the basic estimators of mutual information (Divergence-based, BA, UBA, TUBA, CLUB, FLO, InfoNCE, ...). Use as guide the available code (mi_estimators.py)

- Maybe, one file per demonstration example (multivariate Gaussians, contrastive learning(?), CIFAR10/100, ...)

- Gamma function spaces: continuous & bounded, L-Lipschitz, equivariant, user-defined




HOW TO RUN
----------
python3 1d_Gaussian_demo.py --sample_size 100000 --batch_size 10000 --epochs 200 --method KLD-DV

python 1d_Gaussian_demo.py --sample_size 10000 --batch_size 1000 --epochs 200 --method KLD-DV --framework torch/tf/jax

python N_dim_Gaussian_demo.py --sample_size 10000 --batch_size 1000 --epochs 200 --method KLD-DV-GP --framework torch/tf/jax

python N_dim_Gaussian_demo.py --sample_size 10000 --batch_size 1000 --epochs 200 --method KLD-DV-GP --dimension 2 --framework torch/tf/jax


(NOT READY YET) python3 large_scale_bio_run_new.py --sample_size 100000 --batch_size 10000 --alpha 2.0 --no_repeats 50 --epochs 200 --method IC-rescaled
# python3 aucs_bio_example.py




MUTUAL INFORMATION (later)
--------------------------
MI
    Divergence-based (inherit the respective divergence; eg, MI_KLD_DV inherits from KLD_DV, too)
    BA
    UBA
    TUBA
    CLUB
    FLO
    InfoNCE
    ...
