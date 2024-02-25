from simulate import simulate
from parameter_fitting import fit_with_grid_search, fit_with_random_search, get_best_params_and_ll
from scipy.stats import pearsonr
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from typing import Callable

# def param_recovery(agent_type:str, parameter_space:dict, fit_function:Callable, num_runs:int=20, seed:int=0):
def param_recovery(agent_type:str, parameter_space:dict,  fit_type='grid_search', num_runs:int=20, seed:int=0):
    if fit_type == 'random_search':
        # sample true parameters from the parameter space 
        true_params = {param: parameter_space[param].rvs(size=num_runs) for param in parameter_space.keys()}

        # print('true params:',true_params)
        fitted_params = {param : [] for param in parameter_space.keys()}
        best_LLs = []
        for run in tqdm(range(num_runs), desc='fitting_runs:', total=num_runs):
            params = {param : true_params[param][run] for param in parameter_space.keys()}
            # simulate the data
            data, _ = simulate(agent_type, params=params, seed=seed)
            # fit the model to the data
            fit_results = fit_with_random_search(agent_type, parameter_space, data)
            # get the best parameters
            best_params, best_LL = get_best_params_and_ll(fit_results)
            # store the fitted parameters and the true parameters
            for param in parameter_space.keys():
                fitted_params[param].append(best_params[param])
            best_LLs.append(best_LL)
            # print(f'run: {run}, alpha: {alpha}, beta: {beta}, best_alpha_mb: {best_alpha_mb}, best_beta_mb: {best_beta_mb}')
            
    else:
        true_params = {param : np.random.uniform(np.min(parameter_space[param]), np.max(parameter_space[param]), num_runs)
                    for param in parameter_space.keys()}

        # print('true params:',true_params)
        fitted_params = {param : [] for param in parameter_space.keys()}
        best_LLs = []
        for run in tqdm(range(num_runs), desc='fitting_runs:', total=num_runs):
            params = {param : true_params[param][run] for param in parameter_space.keys()}
            # simulate the data
            data, _ = simulate(agent_type, params=params, seed=seed)
            # fit the model to the data
            fit_results = fit_with_grid_search(parameter_space, data, agent_type=agent_type)
            # get the best parameters
            best_params_idx = np.unravel_index(fit_results.argmax(), fit_results.shape)
            best_params = {param : parameter_space[param][best_params_idx[i]] for i, param in enumerate(parameter_space.keys())}
            # get the best LL and the corresponding parameters
            best_LL = fit_results[best_params_idx]
            # store the fitted parameters and the true parameters
            for param in parameter_space.keys():
                fitted_params[param].append(best_params[param])
            best_LLs.append(best_LL)
            # print(f'run: {run}, alpha: {alpha}, beta: {beta}, best_alpha_mb: {best_alpha_mb}, best_beta_mb: {best_beta_mb}')

    return fitted_params, true_params, best_LLs

def plot_param_recovery(true_params:dict, fitted_params:dict, title='', max_plots_per_row:int=3, save=False, filename:str='plots/param_recovery.png'):

    n_params = len(true_params)

    rows = (n_params - 1) // max_plots_per_row + 1  # ensure at least one row
    cols = min(n_params, max_plots_per_row)  
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows))
    fig.suptitle(title)
    if n_params == 1:
        axes = [axes]  # make sure axes is iterable
    else:
        axes = axes.flatten()

    for i, param in enumerate(true_params.keys()):
        ax = axes[i]
        true_values = np.array(true_params[param])
        fitted_values = np.array(fitted_params[param])
        sns.scatterplot(x=true_values, y=fitted_values, ax=ax)
        ax.set_title(f'{param}')
        ax.set_xlabel('true')
        ax.set_ylabel('fitted')
        
        # plot the identity line based on combined min and max of true and fitted values
        combined_values = np.concatenate([true_values, fitted_values])
        min_val, max_val = combined_values.min(), combined_values.max()
        ax.plot([min_val, max_val], [min_val, max_val], ls="--", c=".3")
        
        # print pearson correlation
        # corr_coef = np.corrcoef(true_values, fitted_values)[0, 1]
        # print(f'Pearson correlation for {param}: {corr_coef:.2f}')
        corr_coef_scipy, p_value = pearsonr(true_values, fitted_values)
        print(f'Pearson correlation for {param}: {corr_coef_scipy:.3f}, p_value: {p_value}')
  
    fig.tight_layout()
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
    
def plot_param_correlation(fitted_params:dict, title='', save=False, filename='plots/recovered_param_correlation.png'):
    # print the correlation uisng scipy pearson
    param_names = list(fitted_params.keys())
    num_params = len(param_names)
        
    if num_params < 2:
        print('Number of parameters less than 2, cannot calculate correlation')
        return
    
    if num_params == 2:
        corr_1 = pearsonr(fitted_params[param_names[0]], fitted_params[param_names[1]])
        print(f'corr_1: {corr_1}')

        fig, ax = plt.subplots(1, figsize=(6, 6))
        fig.suptitle('Correlation between recovered parameters')
        sns.scatterplot(data=fitted_params, x=param_names[0], y=param_names[1], ax=ax)
        ax.set_title(f'{title}_1')
        plt.show()

    if num_params > 2:
        # plot the correlation matrix
        fitted_params_df = pd.DataFrame(fitted_params)
        corr = fitted_params_df.corr()
        fig, ax = plt.subplots(1, figsize=(6, 6))
        fig.suptitle('Correlation between recovered parameters')
        sns.heatmap(corr, annot=True, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
        ax.set_title(f'{title}_2')
        plt.show()

    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)

    

if __name__ == '__main__':
    # test the plotting functions with random data
    true_params = {'alpha': np.random.uniform(0, 1, 20), 
                   'beta': np.random.uniform(0, 1, 20),
                   'gamma': np.random.uniform(0, 1, 20),
                   'epsilon': np.random.uniform(0, 1, 20)}
    fitted_params = {'alpha': np.random.uniform(0, 1, 20),
                        'beta': np.random.uniform(0, 1, 20),
                        'gamma': np.random.uniform(0, 1, 20),
                        'epsilon': np.random.uniform(0, 1, 20)}
    
    plot_param_recovery(true_params, fitted_params, title='test', max_plots_per_row=3, save=False, filename='plots/param_recovery_test.png')
    plot_param_correlation(fitted_params, title='test', save=False, filename='plots/recovered_param_correlation_test.png')
