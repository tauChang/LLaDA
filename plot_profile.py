import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Step 1: Read the CSV data
file_name = "gsm8k.csv"
df = pd.read_csv(file_name)
df.columns = df.columns.str.strip()

# Separate 'llada' and 'llama'
llada_df = df[df['model'] == 'llada'].copy()
llama_df = df[df['model'] == 'llama'].copy()

# Convert num_steps to numeric for sorting and color scaling
llada_df['num_steps'] = pd.to_numeric(llada_df['num_steps'])

# Normalize num_steps for colormap
norm = mcolors.Normalize(vmin=llada_df['num_steps'].min(), vmax=llada_df['num_steps'].max())
cmap = cm.Blues

# Step 2: Plot
fig, ax = plt.subplots()

# Plot llada points with varying color and connect them
llada_df_sorted = llada_df.sort_values(by='num_steps')
for _, row in llada_df_sorted.iterrows():
    color = cmap(norm(row['num_steps']))
    label = f"{row['model']}_{int(row['num_steps'])}steps"
    ax.scatter(row['throughput'], row['accuracy'], label=label, color=color, edgecolors='black', s=50)

# Connect llada points
ax.plot(llada_df_sorted['throughput'], llada_df_sorted['accuracy'], color=cmap(0.8), linewidth=1)

# Plot llama separately
for _, row in llama_df.iterrows():
    label = f"{row['model']}"
    ax.scatter(row['throughput'], row['accuracy'], label=label, color='red', marker='*', s=80)

# Labeling the plot
ax.set_xlabel("Throughput (tokens/sec)")
ax.set_ylabel("Accuracy")
ax.set_title("GSM8K: Throughput vs Accuracy")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{file_name.split('.')[0]}_plot.png", dpi=300)
