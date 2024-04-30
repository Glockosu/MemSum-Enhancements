# Authors: Alex Johannesson and Saumya Shukla
# Description: This script performs document summarization using different models and measures their performance with ROUGE scores. It supports command-line arguments for flexible execution configurations.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Manually inputting data
data = {
    'Model': ['Model A']*9 + ['Model B']*9,
    'Metric': ['ROUGE-1']*3 + ['ROUGE-2']*3 + ['ROUGE-L']*3 + ['ROUGE-1']*3 + ['ROUGE-2']*3 + ['ROUGE-L']*3,
    'Type': ['Precision', 'Recall', 'F1']*6,
    'Score': [
        # Model A Scores
        0.4878, 0.5433, 0.4969,  # ROUGE-1 Precision, Recall, F1
        0.2296, 0.2500, 0.2323,   # ROUGE-2
        0.4118, 0.4107, 0.4003,  # ROUGE-L
        # Model B Scores
        0.5046, 0.5289, 0.4960,  # ROUGE-1
        0.2354, 0.2429, 0.2309,  # ROUGE-2
        0.4164, 0.4339, 0.4082,   # ROUGE-L
    ]
}

# Convert dictionary to DataFrame
# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Set the style
sns.set(style="whitegrid")

# Create a point plot for ROUGE scores
plt.figure(figsize=(12, 8))
point_plot = sns.pointplot(data=df, x='Metric', y='Score', hue='Model', markers=['o', 's'], linestyles=['-', '--'], dodge=True, palette='deep')
point_plot.set_xlabel('ROUGE Metric', fontsize=14)
point_plot.set_ylabel('Score', fontsize=14)
point_plot.set_title('Comparison of ROUGE Metrics by Model', fontsize=16)
point_plot.set_ylim(0, 1)  # Assuming scores are between 0 and 1

# Enhance legend
handles, labels = point_plot.get_legend_handles_labels()
labels = ['Model A', 'Model B']
plt.legend(handles[0:2], labels, title='Model', loc='upper left', bbox_to_anchor=(1, 1))

# Display the plot
plt.show()