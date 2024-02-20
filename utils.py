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


def plot_stay_probability(data, title=""):
    df = data.copy()
    # Convert 'Rewarded' to a string type for clear plotting
    df['Rewarded'] = df['Rewarded'].map({True: 'Rewarded', False: 'Unrewarded'})
    df['Common'] = df['Common'].map({True: 'Common', False: 'Rare'})

    y_limit_min = 0.5 if df['Stay Probability'].min() > 0.5 else df[
                                                                     'Stay Probability'].min() - 0.1

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    # Create the bar plot
    bar = sns.barplot(x='Rewarded', y='Stay Probability', hue='Common',
                      data=df, ax=ax,
                      order=['Rewarded', 'Unrewarded'],
                      hue_order=['Common', 'Rare'])

    # Set the y-axis limit
    ax.set_ylim(y_limit_min, 1)

    ax.set_xlabel('Rewarded', fontsize=15)
    ax.set_ylabel('Stay Probability', fontsize=15)
    ax.set_title('Stay Probability by Reward and Transition Type', fontsize=20)

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

    # Show the plot
    plt.show()

def plot_stay_probabilities(dfs: list[pd.DataFrame], title='', labels: list[str]=None, max_plots_per_row=4):
    sns.set_style("whitegrid")

    n_plots = len(dfs)
    # Calculate the number of rows and columns for the subplot grid
    rows = (n_plots - 1) // max_plots_per_row + 1  # Ensure at least one row
    cols = min(n_plots, max_plots_per_row)  # Max of 4 columns
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 6*rows), sharey=True, sharex=True)
    
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

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def preprocess_human_data(data_df: pd.DataFrame) -> pd.DataFrame:
    data = data_df.copy()
    # infer common transition from the action taken in stage 1 and isHighProbOne/Two
    data['common_transition'] = np.where(data['stepOneChoice'] == 0,
                                            data['isHighProbOne'],
                                            data['isHighProbTwo'])

    # infer the transition to stage 2 state from the action taken in stage 1 and isHighProbOne/Two
    # update the state_transition_to value only if condition is met, else keep it as it is
    data['state_transition_to'] = np.where((data['stepOneChoice'] == 0) & (data['isHighProbOne'] == True),
                                        1,
                                        np.nan)

    data['state_transition_to'] = np.where((data['stepOneChoice'] == 0) & (data['isHighProbOne'] == False),
                                        2,
                                        data['state_transition_to'])

    data['state_transition_to'] = np.where((data['stepOneChoice'] == 1) & (data['isHighProbTwo'] == True),
                                        2,
                                        data['state_transition_to']) 

    data['state_transition_to'] = np.where((data['stepOneChoice'] == 1) & (data['isHighProbTwo'] == False),
                                        1,
                                        data['state_transition_to'])
    # convert to integers
    data['state_transition_to'] = data['state_transition_to'].astype(int)

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


if __name__ == "__main__":
    # Load latest simulated data from csv
    agent_type = "model_based"
    task_df = load_latest_simulated_data(agent_type)
    stay_prob = calculate_stay_probability(task_df)
    plot_stay_probability(stay_prob)

    human_df = pd.read_csv("data/experiment_data.csv")
    human_stay_prob = calculate_stay_probability(human_df, True)
    plot_stay_probability(human_stay_prob)
