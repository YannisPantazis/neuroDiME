<div align="center">
  
# **NeuroDiME**: A Software Library for Neural-based Divergence and Metric Estimation

<img src="https://img.shields.io/badge/Python-3.10-306998">
<img src="https://img.shields.io/badge/Conda-4.12.0-44903d">
<img src="https://img.shields.io/badge/License-MIT-yellow">
</div>

---

### 🔍 **Overview**
**NeuroDiME** is a software library offering neural-based estimation methods for various **divergences** and **integral probability metrics (IPM)**. It supports a wide range of divergences, including several **f-divergences**, and provides an extensive suite of functionalities for estimation and analysis. 

To dive deeper into the documentation, visit our site: [**NeuroDiME Documentation**](https://neurodime.readthedocs.io/en/latest/index.html) (work in progress)

---

## 🌳 **Class Hierarchy**

Our library offers a structured and intuitive class hierarchy for divergence estimation. Below is an illustrative diagram of how different classes interrelate:

![Class Hierarchy](images/class_hierarchy_2.png)

---

## 📋 **Requirements**
We tested our examples with CUDA 12.5 and cuDNN 8.9.2.

All dependencies are specified in the `requirements.txt`. 

### 🔧 **Quick Setup**
To quickly set up your environment, use the following commands:

```bash
# Step 1: Create a new conda environment
conda create --name neurodime_env python=3.10.4

# Step 2: Install all required packages
pip install -r requirements.txt
```

Alternatively, you can install packages individually:
```bash
pip install tensorflow tensorflow_addons torch torchvision torchaudio torchinfo torchmetrics torch-fidelity
pip install jax[cuda12] flax pandas matplotlib scipy tqdm seaborn
```

---

## 💡 **Examples and Use Cases**

Explore the wide range of example implementations available in NeuroDiME:

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11jeQs149iXYmBRM-O0S5fvMkwkj8m1Tv?usp=sharing) **Multivariate Gaussians**: Experiment with different dimensions and correlation coefficients.
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16rC99fr34160PwNfS1b40OYQEllhVmK5?usp=sharing) **Subpopulation Detection**: Analyze real datasets.
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10_jN8qfONkYwZmshocqlwgpNtscB3E5k?usp=sharing) **Image-based Tasks**: Utilize CNN-based models for divergence estimation.
4. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BUOWmrtpaEtNsaqRYLmQTSSMZn-iZAke?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SX5SOAYWKGUBP_BR-vM6XdX8zssXsUDq?usp=sharing) **Generation/GAN**: Generate images using MNIST and CIFAR-10 datasets.

> 💻 *Note*: All examples have been tested on a single GPU (NVIDIA 4070 Super, 16GB). For the pretrained models used in the Generation/GAN examples, please reach out to us. All the Colab Notebooks are implemented with PyTorch, with support for other frameworks coming soon.

---

## 📂 **Core Python Files**

Our implementation is organized into several key files:

- **`Divergences_tf.py`**, **`Divergences_torch.py`**, **`Divergences_jax.py`**: Contain the core implementations for all major divergence families. You can customize these using input arguments for the test functions/discriminators. Find them under the `models` directory.
- **Demonstration Files**: Each demonstration example (e.g., **1D Gaussian**, **Mixture of Gaussians**, **Subpopulation detection**) has a corresponding Python file located in the `tf_demos`, `torch_demos`, and `jax_demos` directories.

NeuroDiME supports various gamma function spaces, including **continuous & bounded**, **L-Lipschitz**, **equivariant**, and **user-defined** functions.

---

## 🚀 **How to Run**

Ready to explore NeuroDiME? Here’s how you can get started with different examples:

```bash
# Example 1: Run an N-dimensional Gaussian with 1 dimension
python N_dim_Gaussian_demo.py --sample_size 10000 --batch_size 1000 --epochs 200 --method KLD-DV --use_GP True --dimension 1

# Example 2: Train a GAN on the MNIST dataset
python mnist_gan.py --method KLD-DV --use_GP True --conditional True

# Example 3: Train a GAN on CIFAR-10
python cifar10_gan.py --method KLD-DV --use_GP True --conditional True

# Example 4: Load a pre-trained GAN model trained on the MNIST dataset
python mnist_gan.py --method KLD-DV --use_GP True --conditional True --load_model True

# Example 5: Run a biological hypothesis test
python Divergence_bio_hypothesis_test_demo.py --p 0.01 --method KLD-DV 
```

---

## 💬 **Support**

If you encounter any issues or have questions, feel free to open an issue on GitHub. We also welcome suggestions and feedback to improve the library!

---

## 📧 **Contact**

For further inquiries, reach out to us at:

- Email: [a.aggelakis@iacm.forth.gr](a.aggelakis@iacm.forth.gr)

---

## 👥 **Contributors**

The development of **NeuroDiME** is based on the research presented in our paper. The contributors are:

- [**Alexandros Angelakis**](https://aangelakis.github.io/) - (University of Crete, IACM-FORTH)
- [**Yannis Pantazis**](https://sites.google.com/site/yannispantazis/) - (University of Crete, IACM-FORTH)
- [**Jeremiah Birrell**](https://scholar.google.co.uk/citations?user=R60hJGUAAAAJ&hl=en) - (Texas State University)
- [**Markos Katsoulakis**](https://scholar.google.com/citations?user=2PpEwFQAAAAJ&hl=el) - (University of Massachusetts)

---

## 📄 **License**

This project is released under the **MIT License**, granting you the freedom to use, modify, and distribute the software. For more details, refer to the `LICENSE` file.

---

Feel free to reach out if you have any questions or suggestions. Enjoy using **NeuroDiME**! 🎉
