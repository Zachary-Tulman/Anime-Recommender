"""
This script processes the raw MAL data for use by the model.
clean_data() cleans the raw user rating data.
map_ids() remaps the data's indices so it can be used for embeddings.
"""

import pandas as pd
import os
import glob
import json

from config import DATA_DIR, DROPPED_PCT_CUTOFF, USER_RATING_CUTOFF, ANIME_RATING_CUTOFF

def clean_data() -> pd.DataFrame:
    """Filters and cleans all user_anime*.csv and saves a 15m sample locally"""
    anime_data = pd.read_csv(f"{DATA_DIR}anime.csv", sep="\t", usecols=["anime_id", "num_episodes", "score", "type", "genres"])
    data_shard_paths = glob.glob(os.path.join(DATA_DIR, "user_anime*.csv"))

    # TODO: Load anime.csv and get the anime_ids dropped by dropna on ["anime_id", "score", "type", "genres"].
    #       Remove rows with those anime_ids from all shards (whereever that's best done)
    #       model anime_ids mapped in map_ids() become new source of truth for show existence
    
    user_rating_counts = pd.Series(dtype=int)
    anime_rating_counts = pd.Series(dtype=int)

    # First pass
    for path in data_shard_paths:
        shard = pd.read_csv(path, sep="\t", usecols=["user_id", "anime_id", "score", "status", "progress"]) \
        .merge(anime_data[["anime_id", "num_episodes"]], on="anime_id", how="left") \
        .dropna(subset=["score"])   # TODO: connected to above (?)
        
        # Only keep completed or dropped w/ >=50% completion
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

        # Same filters as above with minimum rating cutoffs added
        shard["completion_pct"] = shard["progress"] / shard["num_episodes"]
        shard = shard[(shard["user_id"].isin(valid_users)) \
                      & (shard["anime_id"].isin(valid_anime)) \
                      & ((shard["status"] == "completed") | \
                        ((shard["status"] == "dropped") & (shard["completion_pct"] >= DROPPED_PCT_CUTOFF)))]
        
        shard.drop(columns=["status", "progress", "num_episodes", "completion_pct"], inplace=True)
        clean_shards.append(shard)

    result = pd.concat(clean_shards, ignore_index=True)
    # TODO: use all data instead of sample
    result = result.sample(n=15_000_000, random_state=13)
    
    return result

def map_ids(data: pd.DataFrame):
    """Remaps cleaned data to have contiguous indices"""
    ratings = data.copy()

    user_ids = ratings["user_id"].unique()
    user_id_map = {id: i for i, id in enumerate(user_ids)}
    ratings["user_id"] = ratings["user_id"].map(user_id_map)

    anime_ids = ratings["anime_id"].unique()
    anime_id_map = {int(id): i for i, id in enumerate(anime_ids)}
    ratings["anime_id"] = ratings["anime_id"].map(anime_id_map)

    # save ID mappings and ID-mapped data
    with open(f"{DATA_DIR}user_id_map.json", "w") as f:
        json.dump(user_id_map, f)
    with open(f"{DATA_DIR}anime_id_map.json", "w") as f:
        json.dump(anime_id_map, f)
    ratings.to_csv(f"{DATA_DIR}processed_ratings.csv", index=False)

if __name__ == "__main__":
    cleaned_data = clean_data()
    map_ids(cleaned_data)