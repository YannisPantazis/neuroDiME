import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import math


def main():
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between MNIST Digit Distributions Colormap')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                    help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')

    opt = parser.parse_args()
    opt_dict = vars(opt)
    method = opt_dict['method']

    path = f'./MNIST_{method}_divergence_demo_torch/'

    # Reading and storing all the estimates from all the csv files
    csv_files = sorted(glob.glob(path + "*.csv"))
    numbers = []
    
    for file in csv_files:
        divergence = pd.read_csv(file, header=None).iloc[0,0]
        if math.isnan(divergence):
            numbers.append(math.inf)
        else:
            numbers.append(np.log10((np.abs(divergence))))

    numbers = np.array(numbers)
    numbers = numbers.reshape(10,10)

    plt.figure(figsize=(10, 10))
    sns.heatmap(numbers, annot=True, fmt='.2f', linewidth=2, cmap='coolwarm')
    plt.title(f'Colormap of {method} estimates between all pair of digits using Torch')
    plt.savefig(f"colormap_torch_{method}.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
