import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns

# Load data once clearly
file_path = "./error_data.csv"
df_full = pd.read_csv(file_path, sep=';', decimal=',')
df_full['avg_l2_error'] = df_full['avg_l2_error'].str.strip("'").replace('out of memory', np.nan)

df_full["avg_l2_error"] = df_full["avg_l2_error"].str.replace(',', '.').astype(float)

# Explicitly compute resolution parameters
#df_full["dx"] = 2 * df_full["length"] / df_full["N_x"]
#df_full["dt"] = df_full["duration"] / df_full["N_t"]

# Data subsets for plots clearly separated
structure_df = df_full.iloc[1:36]      # rows 2 to 36
resolution_df = df_full.iloc[44:60]    # rows 18 to 25
structure_df, resolution_df
fig = plt.figure(figsize=(14, 6))

# Left plot: Network structure analysis
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(structure_df['width'], structure_df['depth'], structure_df['avg_l2_error'], c='blue', marker='o')
ax1.set_xlabel('Width')
ax1.set_ylabel('Depth')
ax1.set_zlabel('Avg L2 Error')
ax1.set_title('Average L2 Error vs. PINN Structure')

# Right plot: Resolution parameters analysis
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(resolution_df['dx'], resolution_df['dt'], resolution_df['avg_l2_error'], c='purple', marker='o')
ax2.set_xlabel('dx')
ax2.set_ylabel('dt')
ax2.set_zlabel('Avg L2 Error')
ax2.set_title('Average L2 Error vs. dx and dt')

#
plt.tight_layout()
plt.show()


# Assuming `structure_df` contains the relevant data
width = structure_df["width"].values
depth = structure_df["depth"].values
error = structure_df["avg_l2_error"].values

# Create a pivot table for heatmap representation
heatmap_data = pd.DataFrame({'Width': width, 'Depth': depth, 'Error': error})
heatmap_pivot = heatmap_data.pivot_table(index='Width', columns='Depth', values='Error', aggfunc='first')


# Plot heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_pivot, cmap='coolwarm', annot=True, fmt=".3f", linewidths=0.5, square=True, cbar_kws={'label': 'Avg L2 Error', 'shrink':0.7})

# Labels and title
plt.ylabel("Width")
plt.xlabel("Depth")
plt.title("L2 Error Heatmap")

# Show plot
plt.show()


# Assuming `structure_df` contains the relevant data
dx = resolution_df["dx"].values
dt = resolution_df["dt"].values
error = resolution_df["avg_l2_error"].values

# Create a pivot table for heatmap representation
heatmap_data = pd.DataFrame({'dx':dx,'dt':dt, 'Error': error})
heatmap_pivot = heatmap_data.pivot_table(index='dx', columns='dt', values='Error', aggfunc='first')
heatmap_pivot = heatmap_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=False)
# Create an annotation matrix that retains 'OOM' for NaN values
annot_matrix = heatmap_pivot.applymap(lambda x: 'OOM' if pd.isna(x) else f"{x:.3f}")

# Plot heatmap using seaborn
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    heatmap_pivot, 
    cmap='coolwarm', 
    annot=annot_matrix,  # Use the custom annotation matrix
    fmt="",  # Empty string because values are preformatted
    linewidths=0.5, 
    square=True, 
    cbar_kws={'label': 'Avg L2 Error'}
)

for i in range(heatmap_pivot.shape[0]):
    for j in range(heatmap_pivot.shape[1]):
        if pd.isna(heatmap_pivot.iloc[i, j]):  # Check for NaN values
            ax.text(j + 0.5, i + 0.5, 'out of memory', ha='center', va='center', color='black', fontsize=12)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(14)
# Labels and title
plt.ylabel("dx", fontsize=14)
plt.xlabel("dt", fontsize=14)
plt.title("L2 Error Heatmap ")

# Show plot
plt.show()
