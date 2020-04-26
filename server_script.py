import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
sns.set_style('ticks')
from miniscript import *



# Load data
df_fake = pd.read_csv("data/df_fake.csv", index_col = 0)
df_random = pd.read_csv("data/df_random.csv", index_col = 0)
df_all = pd.concat([df_fake, df_random]).reset_index(drop=True)

df_all = pd.concat([df_fake, df_all])
y_true = np.zeros(len(papers_all)).astype(int)
y_true[(len(df_fake)): ] = 1

# Create embeddings
group_list = []
embeddings = generate_pca_reduced_embeddings(feature_names= ['abstract', 'title'], basepath = "models/embeddings_")
X = PCA_transform_loaded(filename = "models/PCA_4k.pickle", embeddings)
X_embedded = X
df = pd.DataFrame(X_embedded)

# Find groups
cluster_groups = scripts.cluster_hdbscan(X)

# Set random vs fake papers from known list
y_true = np.zeros(len(df))
y_true[0: len(df_fake)] = 1
df["y_true"] = y_true
df["group"] = cluster_groups

#Plot
sns.scatterplot(data = df, x = 0, y = 1, hue = "group", palette="Set1", s = 8, alpha = 0.9)
sns.scatterplot(data = df[0: len(df_fake)], x = 0, y = 1, hue = "blue", s = 14, alpha = 0.9)
plt.savefig("figure.pdf")

#Extract group
groups, counts = np.unique(cluster_groups, return_counts = True)
ranked_cluster_ix = np.argsort(counts)

# Return performance
suspect_df = df_all[cluster_groups == groups[ranked_cluster_ix[-2]]]
test_performance(suspect_df["title"], papers_fake)
counts[ranked_cluster_ix][::-1]
