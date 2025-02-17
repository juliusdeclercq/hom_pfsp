# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:32:38 2024

@author: Julius de Clercq
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

from scipy.stats import wilcoxon


#%%

metaheuristics = ["single_ILS", "ILS", "QILS"]

results = {}
for mh in metaheuristics:
    results.update({mh: pd.read_excel(f"Results_{mh}.xlsx")})


def plot_and_save_combined_results(results):
    # Create a directory called "Plots" if it doesn't exist
    if not os.path.exists("Plots"):
        os.makedirs("Plots")

    # Get unique instances from one of the metaheuristics (assuming all have the same instances)
    any_metaheuristic = next(iter(results.values()))
    unique_instances = any_metaheuristic['instance'].unique()

    # Create a figure for makespan boxplots
    fig, axs = plt.subplots(2, 6, figsize=(30, 10))
    fig.suptitle('Makespans', fontsize=16)

    for idx, instance in enumerate(unique_instances):
        # Combine data for the current instance across all metaheuristics
        combined_data = []
        for mh, df in results.items():
            df_instance = df[df['instance'] == instance]
            df_instance['metaheuristic'] = mh  # Add a column for the metaheuristic name
            combined_data.append(df_instance)

        combined_df = pd.concat(combined_data)

        # Boxplot for makespans
        row = idx // 6
        col = idx % 6
        sns.boxplot(x='metaheuristic', y='makespan', data=combined_df, ax=axs[row, col])

        # Set the instance name as the title above each plot
        axs[row, col].set_title(instance, fontsize=12)

        # Remove x and y labels
        axs[row, col].set_xlabel('')
        axs[row, col].set_ylabel('')

    # Save the makespan boxplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('Plots/Makespan_Boxplots.png')
    plt.close()

    # Create a figure for runtime boxplots
    fig, axs = plt.subplots(2, 6, figsize=(30, 10))
    fig.suptitle('Total runtime', fontsize=16)

    for idx, instance in enumerate(unique_instances):
        # Combine data for the current instance across all metaheuristics
        combined_data = []
        for mh, df in results.items():
            df_instance = df[df['instance'] == instance]
            df_instance['metaheuristic'] = mh  # Add a column for the metaheuristic name
            combined_data.append(df_instance)

        combined_df = pd.concat(combined_data)

        # Boxplot for run times
        row = idx // 6
        col = idx % 6
        sns.boxplot(x='metaheuristic', y='run_time', data=combined_df, ax=axs[row, col])

        # Set the instance name as the title above each plot
        axs[row, col].set_title(instance, fontsize=12)

        # Remove x and y labels
        axs[row, col].set_xlabel('')
        axs[row, col].set_ylabel('')

    # Save the runtime boxplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('Plots/Runtime_Boxplots.png')
    plt.close()

    # Create a figure for convergence plots
    fig, axs = plt.subplots(2, 6, figsize=(30, 10))
    fig.suptitle('Makespan per normalized iteration', fontsize=16)

    for idx, instance in enumerate(unique_instances):
        # Plot convergence for each metaheuristic on the same axis
        for mh, df in results.items():
            df_instance = df[df['instance'] == instance]

            # Convert string of lists back to list of lists
            convergence_lists = df_instance['convergence'].apply(ast.literal_eval).values
            convergence_array = np.vstack(convergence_lists)

            # Compute element-wise average of the convergence values
            avg_convergence = np.mean(convergence_array, axis=0)

            # Create the x-axis (from 0 to 1)
            x_vals = np.linspace(0, 1, len(avg_convergence))

            # Line plot for convergence, one for each metaheuristic
            axs[idx // 6, idx % 6].plot(x_vals, avg_convergence, label=mh)

        # Set the instance name as the title above each plot
        axs[idx // 6, idx % 6].set_title(instance, fontsize=12)

        # Remove x and y labels
        axs[idx // 6, idx % 6].set_xlabel('')
        axs[idx // 6, idx % 6].set_ylabel('')

        # Add legend for metaheuristics for the last plot
        axs[idx // 6, idx % 6].legend()

    # Save the convergence plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('Plots/Convergence_Plots.png')
    plt.close()

# Example usage
# plot_and_save_combined_results(results)  # Call this with the actual 'results' dictionary


#%%



metaheuristics = list(results.keys())

# Initialize a dictionary to store the test results
wilcoxon_results = {
    'makespan': {},
    'run_time': {}
}

# Perform the Wilcoxon signed-rank test for makespans and runtimes
for i in range(len(metaheuristics)):
    for j in range(i + 1, len(metaheuristics)):
        mh1 = metaheuristics[i]
        mh2 = metaheuristics[j]

        # Skip the pair single_ILS - QILS
        if (mh1 == "single_ILS" and mh2 == "QILS"):
            continue

        for instance in results[mh1]['instance'].unique():
            # Get makespan and runtime data for the current instance
            makespan1 = results[mh1][results[mh1]['instance'] == instance]['makespan'].values
            makespan2 = results[mh2][results[mh2]['instance'] == instance]['makespan'].values
            runtime1 = results[mh1][results[mh1]['instance'] == instance]['run_time'].values
            runtime2 = results[mh2][results[mh2]['instance'] == instance]['run_time'].values

            # Ensure we have the same number of observations (5 per instance)
            if len(makespan1) == len(makespan2) == 5 and len(runtime1) == len(runtime2) == 5:
                # Check if the differences are not all zero, then perform the Wilcoxon signed-rank test
                if not all(makespan1 == makespan2):
                    _, p_makespan = wilcoxon(makespan1, makespan2)
                    wilcoxon_results['makespan'][(f'{mh1}, {mh2}', instance)] = p_makespan

                if not all(runtime1 == runtime2):
                    _, p_runtime = wilcoxon(runtime1, runtime2)
                    wilcoxon_results['run_time'][(f'{mh1}, {mh2}', instance)] = p_runtime

# Initialize DataFrames for makespan and runtime results
makespan_df = pd.DataFrame()
runtime_df = pd.DataFrame()

# Restructure the dictionaries into DataFrames, storing only the p-values
for (pair, instance), p_makespan in wilcoxon_results['makespan'].items():
    # Use the instance as the index and metaheuristic pair as the column name
    makespan_df.loc[instance, pair] = p_makespan

for (pair, instance), p_runtime in wilcoxon_results['run_time'].items():
    # Use the instance as the index and metaheuristic pair as the column name
    runtime_df.loc[instance, pair] = p_runtime

# Set the instance names as the row index
makespan_df.index.name = 'Instance'
runtime_df.index.name = 'Instance'

# Combine the two DataFrames with a double column index
combined_df = pd.concat([makespan_df, runtime_df], axis=1, keys=['makespan', 'run_time'])

# Save the combined DataFrame to Excel
combined_df.to_excel('Wilcoxon_Combined_Results_pvalues.xlsx')

print("Combined Wilcoxon test p-values (makespans and runtimes) saved to Excel.")



df_bounds = pd.read_excel("Instance_test.xlsx", sheet_name="Instances")


# Loop through each metaheuristic's DataFrame in the results dictionary
for mh, df in results.items():
    # mh = "QILS"
    # df = results[mh]
    # Merge the current DataFrame with df_bounds to get the upper bound for each instance
    df_merged = df.merge(df_bounds[['Instance name', 'Upper bound']],
                         left_on='instance', right_on='Instance name',
                         how='left')

    # Calculate RPD
    df['RPD'] = (df_merged['makespan'] - df_merged['Upper bound']) / df_merged['Upper bound']

    # Optional: You can drop the 'Instance name' and 'Upper bound' columns if you want to clean up
    # df.drop(columns=['Instance name', 'Upper bound'], inplace=True)

arpd_results = {}

# Loop through each metaheuristic's DataFrame in the results dictionary
for mh, df in results.items():
    # Calculate the average RPD for each instance
    arpd = df.groupby('instance')['RPD'].mean().reset_index()

    # Rename the column for clarity
    arpd.rename(columns={'RPD': f'ARPD_{mh}'}, inplace=True)

    # Store the result in the dictionary
    arpd_results[mh] = arpd

# Merge ARPD results from all metaheuristics into a single DataFrame
arpd_combined = arpd_results[metaheuristics[0]]

for mh in metaheuristics[1:]:
    arpd_combined = arpd_combined.merge(arpd_results[mh], on='instance', how='outer')

# Display the combined ARPD DataFrame
print("Average RPD (ARPD) Results:")
print(arpd_combined)

arpd_combined.to_excel("ARPD.xlsx")






