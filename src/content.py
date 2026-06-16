import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data import load_anime

DATA_DIR = "./data/"

data = load_anime()

# multi-hot encodings
genres_encode = data["genres"].str.get_dummies(sep="|")
type_encode = pd.get_dummies(data["type"])

# Pointwise mutual information (PMI)
genre_counts = genres_encode.sum()
genre_prob = genre_counts / len(genres_encode)

co_occurrence_counts = genres_encode.T.dot(genres_encode)
joint_probs = co_occurrence_counts / len(genres_encode)

base_probs = np.outer(genre_prob, genre_prob)

pmi = np.maximum(np.log(joint_probs / base_probs), 0)

def get_similarity(genre_query: pd.DataFrame, type_query: pd.DataFrame, mask = None, n: int = 10, genre_weight: float = 0.775, type_weight: float = 0.225):
    if mask is None:
        mask = pd.Series(True, index=data.index)

    # apply boolean masking for user-led filtering
    masked_genres = genres_encode[mask]
    masked_type = type_encode[mask]
    masked_titles = data["title"][mask]

    enriched_genre_query = genre_query.dot(pmi)
    genre_similarity = cosine_similarity(enriched_genre_query, masked_genres).flatten() * genre_weight

    type_similarity = cosine_similarity(type_query, masked_type).flatten() * type_weight

    similarity = pd.Series(genre_similarity + type_similarity, index=masked_titles).sort_values(ascending=False).head(n)
    
    return similarity