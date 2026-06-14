import matplotlib.pyplot as plt

# Adjusted for paper size
plt.rcParams.update({'font.size': 8})

def main():
    # Data extracted from the table
    alpha = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    accuracy = [66.53, 66.95, 69.49, 66.53, 66.53, 65.68, 65.25, 65.68, 66.1, 64.83]
    precision = [69.62, 69.24, 72.67, 68.75, 69.53, 67.22, 66.98, 68.21, 71.3, 67.97]
    recall = [66.2, 66.83, 68.89, 66.41, 66.54, 65.19, 64.68, 65.3, 66, 64.72]
    f1_score = [67.87, 68.01, 70.73, 67.56, 68, 66.19, 65.81, 66.72, 68.55, 66.31]

    # Create figure with smaller size for paper
    plt.figure(figsize=(7, 3))

    # Plot each metric
    plt.plot(alpha, accuracy, marker='o', linestyle='-', label='Accuracy', color='#1f77b4', linewidth=1)
    plt.plot(alpha, precision, marker='s', linestyle='-', label='Precision', color='#2ca02c', linewidth=1)
    plt.plot(alpha, recall, marker='^', linestyle='-', label='Recall', color='#d62728', linewidth=1)
    plt.plot(alpha, f1_score, marker='d', linestyle='-', label='F1-Score', color='#9467bd', linewidth=1)

    # Highlight best alpha (0.5)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.text(0.52, 72.0, 'Best (λ=0.5)', color='#333333', fontsize=7, style='italic', fontweight='bold')

    # Highlight the specific points for alpha=0.5
    best_idx = alpha.index(0.5)
    plt.scatter([0.5]*4, [accuracy[best_idx], precision[best_idx], recall[best_idx], f1_score[best_idx]], 
                color='black', s=50, zorder=5, label='_nolegend_')

    # Formatting the plot
    plt.xlabel('λ', fontsize=7)
    plt.ylabel('Metric Score (%)', fontsize=7)
    plt.xticks(alpha, fontsize=7)
    plt.yticks(fontsize=7)

    # Enable grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Legend
    plt.legend(loc='best', fontsize=7, framealpha=0.9)

    plt.tight_layout()

    # Save the plot explicitly as well
    output_path = r'e:\project\kltn\code\plot_alpha_experiment_paper.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}")

    # Show the interactive plot
    plt.show()

if __name__ == '__main__':
    main()
