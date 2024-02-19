# simulate data
# (for now from random agent, as test the environment and task implementation)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


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


def calculate_stay_probability(data: pd.DataFrame, human: bool = False) -> pd.DataFrame:
    # get a copy of the data
    tmp_df = data.copy()

    # flag for the repeated action (stage 1) in the following trial
    tmp_df['repeated_action_stage_1'] = tmp_df['stepOneChoice'].shift(1) == tmp_df[
        'stepOneChoice']

    if human:
        tmp_df['common_transition'] = np.where(tmp_df['stepOneChoice'] == 0,
                                               tmp_df['isHighProbOne'],
                                               tmp_df['isHighProbTwo'])

    # stay probabilities based on conditions
    # 2 factors:
    #       rewarded trail ( whether the reward in stage 2 is greater than )
    #       common_transition ( whether the transition from stage 1 to stage 2 is common or rare)
    results = tmp_df.groupby(['reward', 'common_transition'])[
        'repeated_action_stage_1'].mean().reset_index()

    # rename columns for clarity
    results.rename(
        columns={'repeated_action_stage_1': 'Stay Probability', 'reward': 'Rewarded',
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

    return results


def plot_stay_probability(data):
    df = data.copy()
    # Convert 'Rewarded' to a string type for clear plotting
    df['Rewarded'] = df['Rewarded'].map({True: 'Rewarded', False: 'Unrewarded'})
    df['Common'] = df['Common'].map({True: 'Common', False: 'Rare'})

    y_limit_min = 0.5 if df['Stay Probability'].min() > 0.5 else df[
                                                                     'Stay Probability'].min() - 0.1

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.legend(title='Transition', title_fontsize='13', fontsize='12')

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


if __name__ == "__main__":
    # Load latest simulated data from csv
    agent_type = "model_based"
    task_df = load_latest_simulated_data(agent_type)
    stay_prob = calculate_stay_probability(task_df)
    plot_stay_probability(stay_prob)

    human_df = pd.read_csv("data/experiment_data.csv")
    human_stay_prob = calculate_stay_probability(human_df, True)
    plot_stay_probability(human_stay_prob)
