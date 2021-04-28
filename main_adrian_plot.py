import matplotlib.pyplot as plt  
import numpy as np 
import scipy.io as sio
n_modes = 10
path_mat = "/home/awag/Documents/TFG/MAT/"
batch_size = 50
n_batches = 2

def main():

    rec_data_filename = "/home/awag/Documents/TFG/DATA/TFRECORD/D15/EXTRA.mat" # Path where I keep  data with Phi and U_Avg (To reconstruct field)
    data = sio.loadmat(rec_data_filename) # Loading data
    phi = data['phi'] # Needed to reconstruct velocity field
    U_Avg = data['U_Avg'] # Needed to reconstruct velocity field
    print(phi.shape)

    
    modes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X_d2, Y_d2, Z_d2 = load_data("downsample2")
    corr_coeff_d2 = correlation_coefficient(Y_d2,Z_d2)

    X_d4, Y_d4, Z_d4 = load_data("downsample4")
    corr_coeff_d4 = correlation_coefficient(Y_d4,Z_d4)

    X_d8, Y_d8, Z_d8 = load_data("downsample8")
    corr_coeff_d8 = correlation_coefficient(Y_d8,Z_d8)

    #X_d16, Y_d16, Z_d16 = load_data("downsample16")
    #corr_coeff_d16 = correlation_coefficient(Y_d16,Z_d16)

    X_m2, Y_m2, Z_m2 = load_data("meanfilter2")
    corr_coeff_m2 = correlation_coefficient(Y_m2,Z_m2)

    X_m4, Y_m4, Z_m4 = load_data("meanfilter4")
    corr_coeff_m4 = correlation_coefficient(Y_m4,Z_m4)

    X_m8, Y_m8, Z_m8 = load_data("meanfilter8")
    corr_coeff_m8 = correlation_coefficient(Y_m8,Z_m8)

    #X_m16, Y_m16, Z_m16 = load_data("meanfilter16")
    #corr_coeff_m16 = correlation_coefficient(Y_m16,Z_m16)


    fig, axes = plt.subplots(2)
    custom_plot(modes,corr_coeff_d2, axes[0], "Downsample 2")
    #custom_plot(modes,corr_coeff_m2, axes[1], "Mean Filter 2")
    custom_plot(modes,corr_coeff_d4, axes[1], "Downsample 4")
    #custom_plot(modes,corr_coeff_m4, axes[3], "Mean Filter 4")
    #custom_plot(modes,corr_coeff_d8, axes[4], "Downsample 8")
    #custom_plot(modes,corr_coeff_m8, axes[5], "Mean Filter 8")
    #custom_plot(modes,corr_coeff_d16, axes[6])
    #custom_plot(modes,corr_coeff_m16, axes[7])


    fig1, axes = plt.subplots(2,4)
    custom_plot3(Y_d2,Z_d2,axes[0,0], axes[0,1], axes[0,2],axes[0,3], "Downsample 2")
    custom_plot3(Y_d4,Z_d4,axes[1,0], axes[1,1], axes[1,2],axes[1,3], "Downsample 4")
    #custom_plot3(Y_d8,Z_d8,axes[2,0], axes[2,1], axes[2,2],axes[2,3], "Downsample 8")
    #custom_plot3(Y_d16,Z_d16,axes[3,0], axes[3,1], axes[3,2],axes[3,3])
    #custom_plot3(Y_m2,Z_m4,axes[3,0], axes[3,1], axes[3,2],axes[3,3], "Mean Filter 2")
    #custom_plot3(Y_m4,Z_m4,axes[4,0], axes[4,1], axes[4,2],axes[4,3], "Mean Filter 4")
    #custom_plot3(Y_m8,Z_m8,axes[5,0], axes[5,1], axes[5,2],axes[5,3], "Mean_Filter 8")
    #custom_plot3(Y_m16,Z_m16,axes[7,0], axes[7,1], axes[7,2],axes[7,3])




    fig2, axes = plt.subplots(2)
    axes[0].imshow(X_d2[0,0,:,:])
    axes[0].set_title('Downsample 2')
    #axes[1].imshow(X_m2[0,0,:,:])
    #axes[1].set_title('Mean Filter 2')
    axes[1].imshow(X_d4[0,0,:,:])
    axes[1].set_title('Downsample 4')
    #axes[3].imshow(X_m4[0,0,:,:])
    #axes[3].set_title('Mean Filter 4')
    #axes[4].imshow(X_d8[0,0,:,:])
    #axes[4].set_title('Downsample 2')
    #axes[5].imshow(X_m8[0,0,:,:])
    #axes[5].set_title('Mean Filter 8')
    #plt.show()
    print(Y_d2[0,:].shape)

    fig3, axes = plt.subplots(2)
    plot_field(phi[:8192,:n_modes], Y_d4[0,:], axes[0])
    plot_field(phi[:8192,:n_modes], Z_d4[0,:], axes[1])
    plt.show()

    return


def correlation_coefficient(Y,Z):
    corr_coeff = np.zeros((n_modes,1))
    for j in range(n_modes):
        
        psi_estimation = Z[:,j]
        psi_actual = Y[:,j]
        corr_coeff_matrix = np.corrcoef(psi_estimation, psi_actual)
        #print(corr_coeff_matrix)
        corr_coeff[j] = corr_coeff_matrix[0,1]
        

    return corr_coeff


def load_data(model_name):
    filename = f"{path_mat}{model_name}.npz"
    data = np.load(filename)
    X = data['X'] 
    Y = data['Y']
    Z = data['Z']


    return X, Y, Z


def custom_plot(x, y, ax=None, title = "" , **plt_kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(x, y, **plt_kwargs) ## example plot here
    ax.set_title(title)
    ax.set_ylim(0,1)
    return(ax)

def custom_plot2( y, ax=None, title = "" , **plt_kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(y, **plt_kwargs) ## example plot here
    ax.set_title(title)
    return(ax)

def custom_plot3( Y,Z ,ax0=None, ax1=None, ax2=None, ax3=None ,title = "", **plt_kwargs):

    custom_plot2(Z[:,0], ax0, title)
    custom_plot2(Y[:,0], ax0,title,linestyle = '', marker = 'o')
    custom_plot2(Z[:,1], ax1, title)
    custom_plot2(Y[:,1], ax1,"Mode 2",linestyle = '', marker = 'o')
    custom_plot2(Z[:,4], ax2, title)
    custom_plot2(Y[:,4], ax2,"Mode 5",linestyle = '', marker = 'o')
    custom_plot2(Z[:,9], ax3, title)
    custom_plot2(Y[:,9], ax3,"Mode 10",linestyle = '', marker = 'o')
    return ax0, ax1, ax2, ax3

def plot_field(phi, Y, ax):
    Y = np.transpose(Y)
    U_vec = phi @ Y
    U = np.reshape(U_vec,(128,64))
    U = np.transpose(U)

    ax.imshow(U)
    return (ax)





main()



