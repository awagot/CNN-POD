import matplotlib.pyplot as plt  
import numpy as np 
n_modes = 10
path_mat = "/home/awag/Documents/TFG/MAT/"


def main():

    X_d2, Y_d2, Z_d2 = load_data("downsample2")
    corr_coeff_d2 = correlation_coefficient(Y_d2,Z_d2)
    #corr_coeff_m2 = correlation_coefficient(Y_m2, Z_m2)


    return


def correlation_coefficient(Y,Z):

    for j in range(n_modes):
        psi_estimation = result[:,j]
        psi_actual = Y[:,j]
        corr_coeff_matrix = np.corrcoef(psi_estimation, psi_actual)
        corr_coeff[j] = corr_coeff_matrix[0,1]

    return corr_coeff


def load_data(model_name):
    filename = f"{path_mat}{model_name}.npz"
    data = np.load(filename)
    X = data['X'] 
    Y = data['Y']
    Z = data['Z']


    return X, Y, Z





main()



