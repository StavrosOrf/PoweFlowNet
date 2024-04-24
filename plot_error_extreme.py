import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

@plt.style.context('utils/article.mplstyle')
def plot_error_extreme(std_levels, errors, extra_errors=None, label_errors=None, label_extra_errors=None,
                       save_path=None):
    label_errors = label_errors or 'Error'
    label_extra_errors = label_extra_errors or 'Error'
    save_path = save_path or os.path.join(os.getcwd(), 'error_extreme.pdf')
    
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_subplot(111)
    
    ax.plot(std_levels, errors, label=label_errors, marker='s', color='tab:blue')
    standard_std_level = std_levels[std_levels == 0.1]
    standard_erros = errors[std_levels == 0.1]
    ax.plot(standard_std_level, standard_erros, marker='*', color='tab:green')
    if extra_errors is not None:
        ax.plot(std_levels, extra_errors, label=label_extra_errors, marker='s', color='tab:red')
        standard_extra_errors = extra_errors[std_levels == 0.1]
        ax.plot(standard_std_level, standard_extra_errors, marker='*', color='tab:green')
        
    ax.set_xlabel('Standard deviation levels')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    
    ax.legend()
    ax.grid(True)
    
    fig.savefig(save_path)
    return fig

def main():
    std_levels = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
    masked_l2_errors = np.array([0.0156, 0.0193, 0.0469, 0.1408, 0.3894, 1.1864])
    mse_errors = np.array([0.0067, 0.0084, 0.0199, 0.0569, 0.1370, 0.3343])
    
    plot_error_extreme(
        std_levels=std_levels,
        errors=masked_l2_errors,
        extra_errors=mse_errors,
        label_errors='Masked L2 error',
        label_extra_errors='MSE error',
        save_path='error_extreme.pdf',
    )

if __name__ == '__main__':
    main()