import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between MNIST Digit Distributions Colormap')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                    help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')

    opt = parser.parse_args()
    opt_dict = vars(opt)
    method = opt_dict['method']

    path = f'./MNIST_{method}_divergence_demo'

    # Reading and storing all the estimates from all the csv files
    csv_files = glob.glob(path + "/*.csv")

    df_list = (pd.read_csv(file, header=None, index_col=None) for file in csv_files)
    
    df = pd.concat(df_list, ignore_index=True)

    div_values = np.zeros((10, 10))

    # Storing the divergence of each combination in a 10x10 matrix
    i = 0
    for j, divergence in enumerate(df.values):
        div_values[i][j % 10] = np.log10((np.abs(divergence)))

        if (j+1) % 10 == 0:
            i = i + 1

    print(div_values)

    # Plotting the colormap of the estimates
    x = np.arange(0, 10, 1)
    y = np.arange(0, 10, 1)
    plt.imshow(div_values, interpolation='none', cmap=plt.get_cmap('binary'))
    plt.title(f'Colormap of {method} estimates between all pair of digits using PyTorch')
    plt.xticks(x)
    plt.yticks(y)
    cbar = plt.colorbar()
    cbar.set_label(r'$\log_{10}{\left({\left|{estimate}\right|}\right)}$')
    plt.show()


if __name__ == "__main__":
    main()
