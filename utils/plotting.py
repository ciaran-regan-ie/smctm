import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP

sns.set_theme(style="ticks")  # Globally set plotting style
plt.rcParams["axes.xmargin"] = 0  # Globally remove x-axis margins


def plot_metrics(metrics: dict):
	for name, step_value_pairs in metrics.items():
		plt.plot([pair[0] for pair in step_value_pairs], [pair[1] for pair in step_value_pairs], label=name)
		plt.xlabel("epoch")
		plt.ylabel(name)
		plt.savefig(f"{name}.png", bbox_inches="tight")
		plt.close()


def plot_umap(states_list: list[np.array], labels: "np.array | None" = None):
	for i, states in enumerate(states_list):
		# Manipulate dimensions and create labels for plotting
		T, B, D = states.shape
		flattened_states = states.reshape(T * B, D)
		time_indices, batch_indices = np.repeat(np.arange(T), B), np.tile(np.arange(B), T)
		# Fit UMAP to states
		scaled_states = StandardScaler().fit_transform(flattened_states)  # Normalise features to prevent any dimensions from dominating the distance calculation
		embeddings = UMAP().fit_transform(scaled_states)
		# Plot with time indices
		scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=time_indices, cmap="Spectral", alpha=0.8)
		cbar = plt.colorbar(scatter, ticks=np.arange(T))
		cbar.set_label("time")
		plt.gca().set_aspect("equal", "datalim")
		plt.savefig(f"umap_time_{i}.png", bbox_inches="tight")
		plt.close()
		# Plot with labels, if provided
		if labels is not None:
			n_categories = len(np.unique(labels))
			flattened_labels = np.tile(labels.flatten(), T)  # Tiles labels across the time dimension (assumes one label per sequence)
			scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=flattened_labels, cmap="tab10", alpha=0.8)
			cbar = plt.colorbar(scatter, ticks=np.arange(n_categories), boundaries=np.arange(n_categories + 1) - 0.5, spacing="uniform")
			cbar.set_label("category")
			cbar.ax.set_yticklabels(np.arange(n_categories))
			plt.gca().set_aspect("equal", "datalim")
			plt.savefig(f"umap_labels_{i}.png", bbox_inches="tight")
			plt.close()