Title: Neural-based Estimation of Divergences


CLASS HIERARCHY
--------------
Divergence
    KLD_DV
    IPM
	Wasserstein_GP
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
    Sinkhorn_Divergence

    
?? in total    
 
Consider including more divergences: skew KLD and/or skew f-divergences (this is merely a matter of dataset definition - have it as an example), Cauchy-Schwartz divergence, Wasserstein GP (1 and 2 sided), Sinkhorn?

Add GP in f-divergences?

!!!Move inside the class the dataset slicing into mini-batches!!!


MI
    Divergence-based (inherit the respective divergence; eg, MI_KLD_DV inherits from KLD_DV, too)
    BA
    UBA
    TUBA
    CLUB
    FLO
    InfoNCE
    ...


PYTHON FILES
------------
- Divergences.py contains all the basic families of divergences (the test function/discriminator is an input argument)

- One file per demonstration example (1D Gaussian, Mixture of Gaussians/heavy-tailed, subpopulation detection, ...)

- Implement also GANs? (Not sure if we want to...)


- MI.py contains all the basic estimators of mutual information (Divergence-based, BA, UBA, TUBA, CLUB, FLO, InfoNCE, ...). Use as guide the available code (mi_estimators.py)

- One file per demonstration example (multivariate Gaussians, contrastive learning(?), CIFAR10/100, ...)

- Gamma function spaces: continuous & bounded, L-Lipschitz, equivariant, user-defined


HOW TO RUN
----------
python3 1d_Gaussian_demo.py --sample_size 100000 --batch_size 10000 --epochs 200 --method KLD-DV


(NOT READY YET) python3 large_scale_bio_run_new.py --sample_size 100000 --batch_size 10000 --alpha 2.0 --no_repeats 50 --epochs 200 --method IC-rescaled



python3 aucs_bio_example.py