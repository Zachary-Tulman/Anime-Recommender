import pandas as pd
import json

DATA_DIR = "./data/"

def map_ids():
    ratings = pd.read_csv(f"{DATA_DIR}ratings_clean_sampled.csv")

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
    ratings.to_csv(f"{DATA_DIR}ratings_sampled_mapped.csv", index=False)

if __name__ == "__main__":
    map_ids()