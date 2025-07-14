# Gideon Vos 2025
# James Cook University, Australia
# www.linkedin/in/gideonvos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations


def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()
    mae_data = data[data['Metric'] == 'mae'].copy()
    runtime_data = data[data['Metric'] == 'runtime'].copy()
    return mae_data, runtime_data


def calculate_summary_statistics(mae_data, methods, alpha=0.05):
    summary_stats = []
    for dataset in mae_data['Dataset'].unique():
        for method in methods:
            values = mae_data[mae_data['Dataset'] == dataset][method].values
            if len(values) > 1:
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                count = len(values)
                se = std / np.sqrt(count)
                ci = stats.t.interval(1-alpha, df=count-1, loc=mean, scale=se)
                summary_stats.append({
                    'Dataset': dataset,
                    'Method': method,
                    'Mean_MAE': mean,
                    'Std_Dev': std,
                    '95%_CI_Lower': ci[0],
                    '95%_CI_Upper': ci[1],
                    'Count': count
                })
    return pd.DataFrame(summary_stats)


def perform_friedman_tests(mae_data, methods):
    results = []
    for dataset in mae_data['Dataset'].unique():
        groups = [mae_data[mae_data['Dataset'] == dataset][m].values for m in methods]
        stat, p_value = stats.friedmanchisquare(*groups)
        results.append({
            'Dataset': dataset,
            'Friedman_Stat': stat,
            'P_value': p_value,
            'Significant (p<0.05)': p_value < 0.05
        })
    return pd.DataFrame(results)


def perform_pairwise_wilcoxon(mae_data, methods):
    results = []
    for dataset in mae_data['Dataset'].unique():
        dataset_data = mae_data[mae_data['Dataset'] == dataset]
        for m1, m2 in combinations(methods, 2):
            try:
                stat, p_value = stats.wilcoxon(dataset_data[m1], dataset_data[m2])
                results.append({
                    'Dataset': dataset,
                    'Method_1': m1,
                    'Method_2': m2,
                    'P_Value': p_value
                })
            except ValueError:
                continue
    if not results:
        return pd.DataFrame()
    pairwise_df = pd.DataFrame(results)
    p_value_tables = {
        dataset: df.pivot(index='Method_1', columns='Method_2', values='P_Value').round(4)
        for dataset, df in pairwise_df.groupby('Dataset')
    }
    return p_value_tables


def perform_ranking_analysis(mae_data, methods):
    ranks = mae_data[methods].rank(axis=1, method='min')
    avg_ranks = ranks.mean().sort_values()
    return avg_ranks.reset_index().rename(columns={'index': 'Method', 0: 'Average_Rank'})


def analyze_performance_by_group(mae_data, methods, group_by_col):
    return mae_data.groupby(group_by_col)[methods].mean()


def create_visualizations(summary_df, avg_ranks_df, dataset_performance, seq_len_performance, methods):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 20,
        'legend.fontsize': 12
    })
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Comparative Analysis of Regularization Methods', y=1.02)
    palette = sns.color_palette("viridis", n_colors=len(methods))
    sns.barplot(data=summary_df, x='Method', y='Mean_MAE', ax=axes[0, 0], palette=palette, order=methods)
    axes[0, 0].set_title('Overall Performance by Method')
    axes[0, 0].set_ylabel('Mean Absolute Error (MAE)')
    axes[0, 0].set_xlabel('Method')
    axes[0, 0].tick_params(axis='x', rotation=45)
    sns.barplot(data=avg_ranks_df, x='Method', y='Average_Rank', ax=axes[0, 1], palette=palette, order=methods)
    axes[0, 1].set_title('Average Method Rank (Lower is Better)')
    axes[0, 1].set_ylabel('Average Rank')
    axes[0, 1].set_xlabel('Method')
    axes[0, 1].tick_params(axis='x', rotation=45)
    dataset_performance.plot(kind='bar', ax=axes[1, 0], color=palette, width=0.8)
    axes[1, 0].set_title('Mean MAE by Dataset')
    axes[1, 0].set_ylabel('Mean Absolute Error (MAE)')
    axes[1, 0].set_xlabel('Dataset')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    seq_len_performance.plot(kind='line', marker='o', ax=axes[1, 1], color=palette)
    axes[1, 1].set_title('Mean MAE across Sequence Lengths')
    axes[1, 1].set_ylabel('Mean Absolute Error (MAE)')
    axes[1, 1].set_xlabel('Sequence Length')
    axes[1, 1].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def main():
    filepath = 'results csv file here for each method of RNN, LSTM, PATCHTST'
    methods = ['No Dropout', 'Dropout', 'MC Dropout', 'Synaptic Pruning']
    mae_data, _ = load_and_prepare_data(filepath)
    summary_df = calculate_summary_statistics(mae_data, methods)
    friedman_df = perform_friedman_tests(mae_data, methods)
    pairwise_p_values = perform_pairwise_wilcoxon(mae_data, methods)
    avg_ranks_df = perform_ranking_analysis(mae_data, methods)
    dataset_performance = analyze_performance_by_group(mae_data, methods, 'Dataset')
    seq_len_performance = analyze_performance_by_group(mae_data, methods, 'Sequence Length')
    
    print("Summary Statistics for Mean Absolute Error (MAE)")
    print(summary_df.round(4).to_string(index=False))
    print("Friedman Test for Differences in Methods (per Dataset)")
    print(friedman_df.round(6).to_string(index=False))
    print("Pairwise Wilcoxon Signed-Rank Test P-Values (per Dataset)")
    for dataset, p_table in pairwise_p_values.items():
        print(f"\n--- {dataset} Dataset ---")
        print(p_table.fillna('-'))
    print("Average Method Ranks Across All Trials")
    print(avg_ranks_df.round(3).to_string(index=False))
    create_visualizations(
        summary_df=summary_df,
        avg_ranks_df=avg_ranks_df,
        dataset_performance=dataset_performance,
        seq_len_performance=seq_len_performance,
        methods=avg_ranks_df['Method'].tolist() 
    )

if __name__ == '__main__':
    main()