import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def random_walk_gaussian(prob, sd, min_prob=0, max_prob=1):
    new_prob = prob + np.random.normal(scale=sd, size=np.shape(prob))
    new_prob = np.clip(new_prob, min_prob, max_prob)
    return new_prob

# Load latest simulated data from csv
def load_latest_simulated_data(agent_type):
    data_folder = os.path.join("data", "simulated", agent_type)
    timestamped_folders = os.listdir(data_folder)
    timestamped_folders.sort()
    latest_folder = timestamped_folders[-1]
    filename = os.path.join(data_folder, latest_folder, "simulated_data.csv")
    print("Loading data from", filename)
    task_df = pd.read_csv(filename)
    return task_df

def save_simulated_data(task_df: pd.DataFrame, agent_type: str):
    # save the data to a csv file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join("data", "simulated", agent_type, timestamp)
    # Create folder if it does not exist
    os.makedirs(file_path, exist_ok=True)
    filename = os.path.join(file_path, "simulated_data.csv")
    task_df.to_csv(filename, index=False)

def calculate_stay_probability(data: pd.DataFrame) -> pd.DataFrame:
    # get a copy of the data
    tmp_df = data.copy()

    # flag for the repeated action (stage 1), same action as the previous trial
    tmp_df['repeated_stepOneAction'] = tmp_df['stepOneChoice'].shift(1) == tmp_df[
        'stepOneChoice']
    
    # flag for the repeated action (stage 1), same action as the next trial (to calculate the stay probability)
    tmp_df['repeated_stepOneAction_next'] = tmp_df['repeated_stepOneAction'].shift(-1)
    
    # discard last trial (no next trial to compare with)
    tmp_df = tmp_df.iloc[:-1]

    # stay probabilities based on conditions
    # 2 factors:
    #       rewarded trail ( whether the reward in stage 2 is greater than )
    #       common_transition ( whether the transition from stage 1 to stage 2 is common or rare)
    results = tmp_df.groupby(['reward', 'common_transition'])[
        'repeated_stepOneAction_next'].mean().reset_index()

    # rename columns for clarity
    results.rename(
        columns={'repeated_stepOneAction_next': 'Stay Probability', 'reward': 'Rewarded',
                 'common_transition': 'Common'}, inplace=True)

    conditions = {
        (True, True): 'rewarded_common',
        (True, False): 'rewarded_rare',
        (False, True): 'unrewarded_common',
        (False, False): 'unrewarded_rare'
    }
    results['Condition'] = results.apply(
        lambda row: conditions[(row['Rewarded'], row['Common'])], axis=1)

    # rounding the stay probabilities
    results['Stay Probability'] = results['Stay Probability'].apply(
        lambda x: np.round(x, 3))

    return results, tmp_df

def plot_stay_probabilities(dfs: list[pd.DataFrame], title='', labels: list[str]=None, max_plots_per_row=4, save=False, filename="plots/stay_probabilities.png"):
    sns.set_style("whitegrid")

    if isinstance(dfs, pd.DataFrame) or not isinstance(dfs, list):
        dfs = [dfs]  # Wrap the single DataFrame in a list
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    n_plots = len(dfs)
    # Calculate the number of rows and columns for the subplot grid
    rows = (n_plots - 1) // max_plots_per_row + 1  # Ensure at least one row
    cols = min(n_plots, max_plots_per_row)  # Max of 4 columns
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows), sharey=True, sharex=True)
    
    # If there's only one subplot, axes won't be an array, so we wrap it in a list for consistency
    if n_plots == 1:
        axes = [axes]
    else:
        # Flatten the axes array to simplify indexing
        axes = axes.flatten()

    fig.suptitle(title)

    if labels is None:
        labels = [f"Plot {i}" for i in range(len(dfs))]
    if len(labels) < len(dfs):
        labels = labels + [f"Plot {i}" for i in range(len(dfs) - len(labels))]
    
    min_stay_prob = np.min([data['Stay Probability'].min() for data in dfs])
    y_limit_min = 0.5 if min_stay_prob > 0.5 else min_stay_prob - 0.1

    for i, data in enumerate(dfs):
        ax = axes[i]
        df = data.copy()
        # Convert 'Rewarded' to a string type for clear plotting
        df['Rewarded'] = df['Rewarded'].map({True: 'Rewarded', False: 'Unrewarded'})
        df['Common'] = df['Common'].map({True: 'Common', False: 'Rare'})
        # Create the bar plot
        bar = sns.barplot(x='Rewarded', y='Stay Probability', hue='Common',
                        data=df, ax=ax,
                        order=['Rewarded', 'Unrewarded'],
                        hue_order=['Common', 'Rare'])

        # Set the y-axis limit
        ax.set_ylim(y_limit_min, 1)

        ax.set_title(labels[i], fontsize=20)

        # Set the size of the legend and the title of the legend
        ax.legend(title_fontsize='13', fontsize='12')

        # Set the size of the x and y ticks labels
        ax.tick_params(labelsize=12)

        # Add percentages on top of each bar
        for p in bar.patches:
            bar.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points', fontsize=12)
            
        if i >= (n_plots - max_plots_per_row):
            ax.set_xlabel('Reward', fontsize=15)
        else:
            # Hide the x-axis label
            ax.set_xlabel('', visible=False)
        if i%max_plots_per_row == 0:
            ax.set_ylabel('Stay Probability', fontsize=15)
        else:
            # Hide the y-axis label
            ax.set_ylabel('', visible=False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
    
def preprocess_human_data(data_df: pd.DataFrame) -> pd.DataFrame:
    data = data_df.copy()
    # infer common transition from the action taken in stage 1 and isHighProbOne/Two
    data['common_transition'] = np.where(data['stepOneChoice'] == 0,
                                            data['isHighProbOne'],
                                            data['isHighProbTwo'])

    # infer the state transition to from the action taken in stage 2
    data['state_transition_to'] = (data['stepTwoChoice'] // 2) + 1  # 1 if choice is 0 or 1. 2 if choice is 2 or 3

    # convert the rewardProbabilities to a list
    data['rewardProbabilities'] = data['rewardProbabilities'].apply(
        lambda x: np.array([float(i) for i in x.strip('[]').replace(',', ' ').split()]))
    
    # convert stepTwoChoice from range 0-3 to 0-1
    data['stepTwoChoice'] = data['stepTwoChoice'] % 2

    return data

def print_simple_statistics(data: pd.DataFrame, full=False, title=""):
    # print some statistics 
    # print_simple_statistics(task_df)
    task_df = data.copy()
    print("###", title)

    print("common transitions percentage:", np.mean(task_df["common_transition"])*100, "%")
    print("rewarded trails percentage:", np.mean(task_df["reward"] > 0)*100, "%")

    if full:
        print("transition percentage from state 0 action 0 to state 1:", np.mean(task_df[task_df["stepOneChoice"] == 0]["state_transition_to"] == 1)*100, "%")
        print("transition percentage from state 0 action 1 to state 2:", np.mean(task_df[task_df["stepOneChoice"] == 1]["state_transition_to"] == 2)*100, "%")
        # get the counts of state transitions and action in stage 1
        counts_stage_2_action_1 = pd.DataFrame({
            'State Counts': task_df['state_transition_to'].value_counts(),
            'Action Stage 1 Counts': task_df['stepOneChoice'].value_counts()
        })
        display(counts_stage_2_action_1)

        mean_reward = task_df.groupby('state_transition_to')['reward'].mean().reset_index()
        mean_reward.columns = ['state_transition_to', 'mean_reward']
        display(mean_reward)

        # get the reward probability distributions for the final stage (2)
        def index_reward_probabilities(row):
            try:
                return row['rewardProbabilities'][row['state_transition_to'] + row['stepTwoChoice']]
            except:
                print(row)

        task_df_tmp = task_df.copy()
        task_df_tmp['mean_reward_prob_stepTwoChoice'] = task_df.apply(index_reward_probabilities, axis=1)
        reward_probabilities = task_df_tmp.groupby(['state_transition_to', 'stepTwoChoice'])['mean_reward_prob_stepTwoChoice'].mean().reset_index()
        display(reward_probabilities)

def calculate_running_step_probabilities(data):
    task_df = data.copy()
    # Initialize columns for stay decisions and running probabilities
    task_df['stay_decision'] = False
    task_df['common_rewarded_prob'] = 0.0
    task_df['common_unrewarded_prob'] = 0.0
    task_df['rare_rewarded_prob'] = 0.0
    task_df['rare_unrewarded_prob'] = 0.0

    # Trackers for calculating running probabilities
    stay_counts = {'common_rewarded': 0, 'common_unrewarded': 0, 'rare_rewarded': 0, 'rare_unrewarded': 0}
    total_counts = {'common_rewarded': 0, 'common_unrewarded': 0, 'rare_rewarded': 0, 'rare_unrewarded': 0}

    for i in range(1, len(task_df)):
        current = task_df.iloc[i]
        prev = task_df.iloc[i-1]

        # Check if the participant stayed with the same choice
        if current['stepOneChoice'] == prev['stepOneChoice']:
            task_df.loc[i, 'stay_decision'] = True

        condition = ('common_' if current['common_transition'] else 'rare_') + ('rewarded' if current['reward'] else 'unrewarded')

        # Update counts
        if task_df.loc[i, 'stay_decision']:
            stay_counts[condition] += 1
        total_counts[condition] += 1

        # Calculate running probabilities
        for key in stay_counts:
            if total_counts[key] > 0:
                task_df.loc[i, key + '_prob'] = stay_counts[key] / total_counts[key]

    return task_df

def plot_running_step_probabilities(task_dfs:list, labels:list=None,window_size=1, max_plots_per_row=3, title='', save=False, filename="plots/running_step_probabilities.png"):
    
    if isinstance(task_dfs, pd.DataFrame) or not isinstance(task_dfs, list):
        task_dfs = [task_dfs]
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    # Create a copy of the DataFrame to avoid modifying the original
    n_plots = len(task_dfs)
    # Calculate the number of rows and columns for the subplot grid
    rows = (n_plots - 1) // max_plots_per_row + 1  # Ensure at least one row
    cols = min(n_plots, max_plots_per_row)  # Max of 4 columns
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows), sharey=True, sharex=True)
    
    # If there's only one subplot, axes won't be an array, so we wrap it in a list for consistency
    if n_plots == 1:
        axes = [axes]
    else:
        # Flatten the axes array to simplify indexing
        axes = axes.flatten()

    fig.suptitle(title)

    if labels is None:
        labels = [f"Plot {i}" for i in range(len(task_dfs))]
    if len(labels) < len(task_dfs):
        labels = labels + [f"Plot {i}" for i in range(len(task_dfs) - len(labels))]
    
    for i, data in enumerate(task_dfs):
        ax = axes[i]
        df_copy = data.copy()

        # Calculate moving averages on the copy
        df_copy['common_rewarded_prob_ma'] = df_copy['common_rewarded_prob'].rolling(window=window_size, min_periods=1).mean()
        df_copy['common_unrewarded_prob_ma'] = df_copy['common_unrewarded_prob'].rolling(window=window_size, min_periods=1).mean()
        df_copy['rare_rewarded_prob_ma'] = df_copy['rare_rewarded_prob'].rolling(window=window_size, min_periods=1).mean()
        df_copy['rare_unrewarded_prob_ma'] = df_copy['rare_unrewarded_prob'].rolling(window=window_size, min_periods=1).mean()

        # Plot each condition's moving average from the copied DataFrame
        ax.plot(df_copy['trial_index'], df_copy['common_rewarded_prob_ma'], label='Common Rewarded (MA)', linestyle='-', color='b')
        ax.plot(df_copy['trial_index'], df_copy['common_unrewarded_prob_ma'], label='Common Unrewarded (MA)', linestyle='--', color='b')
        ax.plot(df_copy['trial_index'], df_copy['rare_rewarded_prob_ma'], label='Rare Rewarded (MA)', linestyle='-', color='orange')
        ax.plot(df_copy['trial_index'], df_copy['rare_unrewarded_prob_ma'], label='Rare Unrewarded (MA)', linestyle='--', color='orange')

        ax.set_title(labels[i])
        ax.legend()
        ax.grid(True)

        if i >= (n_plots - max_plots_per_row):
            ax.set_xlabel('Trial Index', fontsize=15)
        else:
            # Hide the x-axis label
            ax.set_xlabel('', visible=False)
        if i%max_plots_per_row == 0:
            ax.set_ylabel('Running Stay Probability (MA)', fontsize=15)
        else:
            # Hide the y-axis label
            ax.set_ylabel('', visible=False)

    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
    
def softmax(arr, beta):
    e_x = np.exp(beta * (arr - np.max(arr)))  # subtract max value to prevent overflow
    return e_x / e_x.sum(axis=0)  # axis=0 for column-wise operation if arr is 2D, otherwise it's not needed

def calculate_bic(num_params, num_data_points, ll):
    """
    Calculates Bayesian Information Criterion to be used in model comparison
    :param num_params: Number of free parameters that the model has
    :param num_data_points: Number of data points the model has been fitted to
    :param ll: Maximum log likelihood estimation for the model given data
    :return:
    """
    return num_params * np.log(num_data_points) - 2 * ll

# def plot_stay_probability(data, title="", save=False, filename="plots/stay_probabilities.png"):
#     df = data.copy()
#     # Convert 'Rewarded' to a string type for clear plotting
#     df['Rewarded'] = df['Rewarded'].map({True: 'Rewarded', False: 'Unrewarded'})
#     df['Common'] = df['Common'].map({True: 'Common', False: 'Rare'})

#     y_limit_min = 0.5 if df['Stay Probability'].min() > 0.5 else df[
#                                                                      'Stay Probability'].min() - 0.1

#     sns.set_style("whitegrid")

#     fig, ax = plt.subplots(figsize=(10, 6))
#     fig.suptitle(title)

#     # Create the bar plot
#     bar = sns.barplot(x='Rewarded', y='Stay Probability', hue='Common',
#                       data=df, ax=ax,
#                       order=['Rewarded', 'Unrewarded'],
#                       hue_order=['Common', 'Rare'])

#     # Set the y-axis limit
#     ax.set_ylim(y_limit_min, 1)

#     ax.set_xlabel('Rewarded', fontsize=15)
#     ax.set_ylabel('Stay Probability', fontsize=15)
#     ax.set_title('Stay Probability by Reward and Transition Type', fontsize=20)

#     # Set the size of the legend and the title of the legend
#     ax.legend(title_fontsize='13', fontsize='12')

#     # Set the size of the x and y ticks labels
#     ax.tick_params(labelsize=12)

#     # Add percentages on top of each bar
#     for p in bar.patches:
#         bar.annotate(format(p.get_height(), '.2f'),
#                      (p.get_x() + p.get_width() / 2., p.get_height()),
#                      ha='center', va='center',
#                      xytext=(0, 10),
#                      textcoords='offset points', fontsize=12)

#     fig.tight_layout()
#     plt.show()
#     if save:
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
#         fig.savefig(filename)

# def plot_running_step_probabilities(task_df, window_size=1, title='', save=False, filename="plots/running_step_probabilities.png"):
#     # Create a copy of the DataFrame to avoid modifying the original
#     df_copy = task_df.copy()

#     # Calculate moving averages on the copy
#     df_copy['common_rewarded_prob_ma'] = df_copy['common_rewarded_prob'].rolling(window=window_size, min_periods=1).mean()
#     df_copy['common_unrewarded_prob_ma'] = df_copy['common_unrewarded_prob'].rolling(window=window_size, min_periods=1).mean()
#     df_copy['rare_rewarded_prob_ma'] = df_copy['rare_rewarded_prob'].rolling(window=window_size, min_periods=1).mean()
#     df_copy['rare_unrewarded_prob_ma'] = df_copy['rare_unrewarded_prob'].rolling(window=window_size, min_periods=1).mean()

#     fig, ax = plt.subplots(figsize=(12, 8))
#     fig.suptitle(title)
#     # Plot each condition's moving average from the copied DataFrame
#     ax.plot(df_copy['trial_index'], df_copy['common_rewarded_prob_ma'], label='Common Rewarded (MA)', linestyle='-', color='b')
#     ax.plot(df_copy['trial_index'], df_copy['common_unrewarded_prob_ma'], label='Common Unrewarded (MA)', linestyle='--', color='b')
#     ax.plot(df_copy['trial_index'], df_copy['rare_rewarded_prob_ma'], label='Rare Rewarded (MA)', linestyle='-', color='orange')
#     ax.plot(df_copy['trial_index'], df_copy['rare_unrewarded_prob_ma'], label='Rare Unrewarded (MA)', linestyle='--', color='orange')

#     ax.set_title('Running Step Probabilities Over Trials (Moving Average)')
#     ax.set_xlabel('Trial Index')
#     ax.set_ylabel('Running Stay Probability (MA)')
#     ax.legend()
#     ax.grid()

#     plt.show()
#     if save:
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
#         fig.savefig(filename)