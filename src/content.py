import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "./data/"

data = pd.read_csv(f"{DATA_DIR}anime.csv", sep="\t", usecols=["anime_id", "title", "genres", "type"])

genre_encode = data["genres"].str.get_dummies(sep="|")
type_encode = pd.get_dummies(data["type"])

features = pd.concat([genre_encode, type_encode], axis=1)

def get_similarity(query: pd.DataFrame, features: pd.DataFrame, n: int = 10):
    similarity = cosine_similarity(query, features).flatten()
    similarity = pd.Series(similarity, index=data["anime_id"]).sort_values(ascending=False).head(n)
    return similarity