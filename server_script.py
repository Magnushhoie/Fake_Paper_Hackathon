import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
sns.set_style('ticks')
from miniscript import *

# Interface with BioBERT server
from bert_serving.client import BertClient
bc = BertClient()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_embeddings(sentences):
    bc_sentences = list(sentences)
    
    output = []
    for i, bc_sub_list in enumerate(chunks(bc_sentences, 1000)):
        print(i*1000, "/", len(bc_sentences))
        bc_output = bc.encode(bc_sub_list)
        for s in bc_output: output.append(s)
    return(bc_output)

# Load data
df_fake = pd.read_csv("data/df_fake.csv", index_col = 0)
df_random = pd.read_csv("data/df_random.csv", index_col = 0)
df_all = pd.read_csv("data/df_all.csv", index_col = 0)
#df_all = pd.concat([df_fake, df_random]).reset_index(drop=True)

#df_all = pd.concat([df_fake, df_all])
y_true = np.zeros(len(df_all)).astype(int)
y_true[0:(len(df_fake))] = 1

# Create embeddings
group_list = []
embeddings = generate_embeddings(feature_names= ['abstract', 'title'], basepath = "models/embeddings_")
X = PCA_transform_loaded(filename = "models/PCA_4k.pickle", X = embeddings)
X_embedded = X
df = pd.DataFrame(X_embedded)

# Find groups
cluster_groups = cluster_hdbscan(X)

# Set random vs fake papers from known list
y_true = np.zeros(len(df))
y_true[0: len(df_fake)] = 1
df["y_true"] = y_true.astype(int)
df["group"] = cluster_groups.astype(int)

#Plot
sns.scatterplot(data = df, x = 0, y = 1, hue = "group", palette="Set1", s = 8, alpha = 0.9)
sns.scatterplot(data = df[0: len(df_fake)], x = 0, y = 1, color = "blue", s = 14, alpha = 0.9)
plt.savefig("figure.pdf")

#Extract group
groups, counts = np.unique(cluster_groups, return_counts = True)
ranked_cluster_ix = np.argsort(counts)

# Return performance
#suspect_df = df_all[cluster_groups == groups[ranked_cluster_ix[-2]]]
#test_performance(suspect_df["title"], papers_fake)
counts[ranked_cluster_ix][::-1]

# Save output df
print(df.shape, df_all.shape)
df[['doi', 'authors', 'journal', 'year', 'title', 'abstract']] = df_all[['doi', 'authors', 'journal', 'year', 'title', 'abstract']]
df.to_csv("df_output.csv")