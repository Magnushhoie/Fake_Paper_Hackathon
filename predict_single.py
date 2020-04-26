import numpy as np
import pickle

import argparse
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('title', type=str, help='Title')
parser.add_argument('abstract', type=str, help='Abstract')
args = parser.parse_args()
title, abstract = args.title, args.abstract

# Interface with BioBERT server
from bert_serving.client import BertClient
bc = BertClient()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_embeddings(sentences):
    bc_sentences = [sentences]
    bc_output = bc.encode(bc_sentences)
    return(bc_output)

def generate_embeddings_single(title, abstract):
    embedding_list = []
    
    for sentences in [title, abstract]:
        embeddings = get_embeddings(sentences)
        embedding_list.append(embeddings)
        
    embeddings_final = np.mean(embedding_list, axis = 0)
    return(embeddings_final)

embeddings = generate_embeddings_single(title, abstract)
pca = pickle.load( open( "models/PCA_4k.pickle", "rb" ) )
X = pca.transform(embeddings)
print(X)


