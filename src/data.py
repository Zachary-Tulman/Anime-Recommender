"""
This file contains data access functions
"""

import pandas as pd

from config import DATA_DIR

def load_anime() -> pd.DataFrame:
    """Loads anime.csv into a dataframe for use by filters.py:build_mask()"""
    anime_data = pd.read_csv(f"{DATA_DIR}anime.csv", sep="\t", usecols=["anime_id", "title", "genres", "type", \
                                                                         "score", "status", "num_episodes", "score_count"])
    anime_data = anime_data.dropna(subset=["anime_id", "score", "type", "genres"])

    return anime_data

def load_processed_ratings() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}processed_ratings.csv")