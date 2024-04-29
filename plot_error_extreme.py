import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

@plt.style.context('utils/article.mplstyle')
def plot_error_extreme(std_levels, errors, extra_errors=None, label_errors=None, label_extra_errors=None,
                       save_path=None, twinx=False):
    label_errors = label_errors or 'Error'
    label_extra_errors = label_extra_errors or 'Error'
    save_path = save_path or os.path.join(os.getcwd(), 'error_extreme.pdf')
    
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_subplot(111)
    
    h0 = ax.plot(std_levels, errors, label=label_errors, marker='s', color='tab:blue')
    standard_std_level = std_levels[std_levels == 0.1]
    standard_erros = errors[std_levels == 0.1]
    ax.plot(standard_std_level, standard_erros, marker='*', color='tab:green')
    if extra_errors is not None:
        _ax = ax.twinx() if twinx else ax
        h1 = _ax.plot(std_levels, extra_errors, label=label_extra_errors, marker='s', color='tab:red')
        standard_extra_errors = extra_errors[std_levels == 0.1]
        _ax.plot(standard_std_level, standard_extra_errors, marker='*', color='tab:green')
        _ax.set_ylabel(label_extra_errors)
        _ax.legend(loc='upper right')
        # _ax.set_yscale('log')
        
    ax.set_xlabel('Standard Deviation Levels')
    if not twinx:
        ax.set_ylabel('Error')
        ax.grid(True)
    else:
        ax.set_ylabel(label_errors)
    # ax.set_yscale('log')
    
    ax.legend(loc='upper left')
    
    fig.savefig(save_path)
    return fig

def main():
    std_levels = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    masked_l2_errors = np.array([0.0159, 0.0244, 0.0667, 0.2168, 0.5242, 1.0747])
    phys_errors = np.array([0.7883, 1.0601, 1.1612, 1.9435, 4.7033, 10.8349])
    
    de_vm_errors = np.array([3e-6, 5e-6, 9e-6, 1.8e-6, 1.8e-5, 3.4e-5]) 
    de_vm_errors = np.sqrt(de_vm_errors) # Vm (pu)
    de_va_errors = np.array([0.8614, 1.6664, 6.8773, 23.6967, 59.6466, 124.8133]) 
    de_va_errors = np.sqrt(de_va_errors) # Va (deg)
    
    # plot_error_extreme(
    #     std_levels=std_levels,
    #     errors=masked_l2_errors,
    #     extra_errors=phys_errors,
    #     label_errors='Masked L2 error',
    #     label_extra_errors='Physical error',
    #     save_path='error_extreme.pdf',
    # )
    plot_error_extreme(
        std_levels=std_levels,
        errors=de_vm_errors,
        extra_errors=de_va_errors,
        label_errors='Vm RMSE (pu)',
        label_extra_errors='Va RMSE (deg)',
        save_path='error_extreme_vm_va.pdf',
        twinx=True,
    )

if __name__ == '__main__':
    main()