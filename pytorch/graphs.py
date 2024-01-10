import ast
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

directions = ["eng-grn", "grn-eng"]
algorithms = ["BPE", "BPE_DROPOUT", "UNIGRAM"]
vocabs = ["250-250", "250-500", "500-250", "500-500"]

data = {direction: {algorithm: {vocab: [] for vocab in vocabs} for algorithm in algorithms} for direction in directions}

for direction in directions:
    for algorithm in algorithms:
        for vocab in vocabs:
            filepath = f"translation_cache/{direction}/{algorithm}/{vocab}/results.txt"
            file1 = open(filepath, 'r')
            Lines = file1.readlines()

            count = 1
            for line in Lines:
                count %= 4
                if count == 2:
                    res = ast.literal_eval(line)
                    data[direction][algorithm][vocab].append(res["score"])
                count+=1

mpl.style.use('seaborn-v0_8')

# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Define x-axis labels
num_epochs_labels = list(range(1, 51, 5))

# Plot for 'eng-grn'
for i, (vocab_category, vocab_data) in enumerate(data['eng-grn'].items()):
    row = i // 3
    col = i % 3
    for algorithm, values in vocab_data.items():
        sns.lineplot(x=num_epochs_labels, y=values, label=f'{algorithm} - {vocab_category}', ax=axes[row, col])

    axes[row, col].set_title(f'eng-grn - {vocab_category}')
    axes[row, col].set_xlabel('Num Epochs')
    axes[row, col].set_ylabel('CHRF Scores')
    axes[row, col].legend()

# Plot for 'grn-eng'
for i, (vocab_category, vocab_data) in enumerate(data['grn-eng'].items()):
    row = (i + 3) // 3
    col = (i + 3) % 3
    for algorithm, values in vocab_data.items():
        sns.lineplot(x=num_epochs_labels, y=values, label=f'{algorithm} - {vocab_category}', ax=axes[row, col])

    axes[row, col].set_title(f'grn-eng - {vocab_category}')
    axes[row, col].set_xlabel('Num Epochs')
    axes[row, col].set_ylabel('CHRF Scores')
    axes[row, col].legend()

plt.tight_layout()

# Save the plots as image files
plt.savefig('seaborn_plots_2x3_chrf_scores.png')


fix, axes = None, None
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, vocab_size in enumerate(vocabs):
    # Plot for 'eng-grn'
    for j, (translation, translation_data) in enumerate(data.items()):
        row = 0 if j == 0 else 1
        col = i
        for algorithm in algorithms:
            values = data[translation][algorithm][vocab_size]
            sns.lineplot(x=num_epochs_labels, y=values, label=f'{algorithm}', ax=axes[row, col])

        axes[row, col].set_title(f'{translation} - {vocab_size}')
        axes[row, col].set_xlabel('Num Epochs')
        axes[row, col].set_ylabel('CHRF Scores')
        axes[row, col].legend()

plt.tight_layout()

# Save the plots as image files
plt.savefig('seaborn_plots_2x4_algorithm_lines.png')