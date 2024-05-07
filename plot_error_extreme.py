import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

@plt.style.context('utils/small_fig_v2.mplstyle')
def plot_error_extreme(std_levels, errors, extra_errors=None, label_errors=None, label_extra_errors=None,
                       save_path=None, twinx=False, title=None):
    label_errors = label_errors or 'Error'
    label_extra_errors = label_extra_errors or 'Error'
    save_path = save_path or os.path.join(os.getcwd(), 'error_extreme.pdf')
    plt.rcParams['text.usetex'] = True
    fig = plt.figure()
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
        _ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15, 1, 1))
        # _ax.set_yscale('log')
        
    ax.set_xlabel('Standard Deviation Levels')
    if not twinx:
        ax.set_ylabel('Error')
    else:
        ax.set_ylabel(label_errors)
    # ax.set_yscale('log')
    
    ax.legend(loc='upper left')
    if title is not None:
        ax.set_title(title)
    
    ax.grid(True)
    
    fig.savefig(save_path)
    return fig

def main():
    std_levels = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    # CASE 14
    # de_vm_errors = np.array([2e-6, 2e-6, 3e-6, 5e-6, 9e-6, 1.5e-5])
    # de_vm_errors = np.sqrt(de_vm_errors) # Vm (pu)
    # de_va_errors = np.array([0.034828, 0.072657, 0.276021, 0.774585, 1.700192, 3.237581])
    # de_va_errors = np.sqrt(de_va_errors) # Va (deg)
    
    # CASE 118
    # masked_l2_errors = np.array([0.0159, 0.0244, 0.0667, 0.2168, 0.5242, 1.0747])
    # phys_errors = np.array([0.7883, 1.0601, 1.1612, 1.9435, 4.7033, 10.8349])
    
    de_vm_errors = np.array([2e-6, 3e-6, 4e-6, 7e-6, 1.6e-5, 3.2e-5]) 
    de_vm_errors = np.sqrt(de_vm_errors) # Vm (pu)
    de_va_errors = np.array([0.7787, 1.2852, 6.4346, 24.3934, 60.2678, 117.9775]) 
    de_va_errors = np.sqrt(de_va_errors) # Va (deg)
    
    # CASE 6470rte
    # de_vm_errors = np.array([0.000026, 0.000031, 0.000052, 0.000112, 0.000235, 0.000379])
    # de_vm_errors = np.sqrt(de_vm_errors) # Vm (pu)
    # de_va_errors = np.array([12.61, 39.4316, 272.9134, 1024.9236, 2492.5876, 3475.6005])
    # de_va_errors = np.sqrt(de_va_errors) # Va (deg)
    
    
    # plot_error_extreme(
    #     std_levels=std_levels,
    #     errors=masked_l2_errors,
    #     extra_errors=phys_errors,
    #     label_errors='Masked L2 error',
    #     label_extra_errors='Physical error',
    #     save_path='error_extreme_ml2_phys.pdf',
    # )
    plot_error_extreme(
        std_levels=std_levels,
        errors=de_vm_errors,
        extra_errors=de_va_errors,
        label_errors='$V^m$ RMSE (p.u.)',
        label_extra_errors=r'$\theta$ RMSE (deg)',
        save_path='error_extreme_vm_va.pdf',
        twinx=True,
        # title='Case 6470rte'
    )

if __name__ == '__main__':
    main()