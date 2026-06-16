import pandas as pd
import os
import glob

USER_RATING_CUTOFF = 25
ANIME_RATING_CUTOFF = 50
DROPPED_PCT_CUTOFF = 0.5
DATA_DIR = "./data/"

def clean_data():
    anime_data = pd.read_csv(f"{DATA_DIR}anime.csv", sep="\t", usecols=["anime_id", "num_episodes"])
    data_shard_paths = glob.glob(os.path.join(DATA_DIR, "user_anime*.csv"))
    
    user_rating_counts = pd.Series(dtype=int)
    anime_rating_counts = pd.Series(dtype=int)

    # First pass
    for path in data_shard_paths:
        shard = pd.read_csv(path, sep="\t", usecols=["user_id", "anime_id", "score", "status", "progress"]) \
        .merge(anime_data, on="anime_id", how="left") \
        .dropna(subset=["score"])
        
        shard["completion_pct"] = shard["progress"] / shard["num_episodes"]
        shard = shard[(shard["status"] == "completed") | \
                      ((shard["status"] == "dropped") & (shard["completion_pct"] >= DROPPED_PCT_CUTOFF))]

        user_rating_counts = user_rating_counts.add(shard["user_id"].value_counts(), fill_value=0)
        anime_rating_counts = anime_rating_counts.add(shard["anime_id"].value_counts(), fill_value=0)

    valid_users = user_rating_counts[user_rating_counts >= USER_RATING_CUTOFF].index
    valid_anime = anime_rating_counts[anime_rating_counts >= ANIME_RATING_CUTOFF].index

    # Second pass
    clean_shards = []
    
    for path in data_shard_paths:
        shard = pd.read_csv(path, sep="\t", usecols=["user_id", "anime_id", "score", "status", "progress"]) \
        .merge(anime_data, on="anime_id", how="left") \
        .dropna(subset=["score"])

        shard["completion_pct"] = shard["progress"] / shard["num_episodes"]
        shard = shard[(shard["user_id"].isin(valid_users)) \
                      & (shard["anime_id"].isin(valid_anime)) \
                      & ((shard["status"] == "completed") | \
                        ((shard["status"] == "dropped") & (shard["completion_pct"] >= DROPPED_PCT_CUTOFF)))]
        
        shard.drop(columns=["status", "progress", "num_episodes", "completion_pct"], inplace=True)
        clean_shards.append(shard)

    result = pd.concat(clean_shards, ignore_index=True)
    result = result.sample(n=15_000_000, random_state=13)
    result.to_csv(f"{DATA_DIR}ratings_clean_sampled.csv", index=False)

def load_anime():
    anime_data = pd.read_csv(f"{DATA_DIR}anime.csv", sep="\t", usecols=["anime_id", "title", "genres", "type", \
                                                                         "score", "status", "num_episodes", "source_type", "score_count"])
    anime_data = anime_data.dropna(subset=["anime_id", "score", "type", "genres"])

    return anime_data

if __name__ == "__main__":
    clean_data()