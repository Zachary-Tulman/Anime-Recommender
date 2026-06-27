"""
Combine Collaborative Filtering, Content-Based Filtering, and Community Score into one recommendation score and rank scores
"""

import pandas as pd
import json
import torch

from config import ANIME_EMBED_DIM, DATA_DIR, ROOT_DIR, USER_EMBED_DIM
from model import AnimeRecommender

def recommend(user_id: str, genres: list[str] | None = None, air_type: str | None = None, min_score: int | None = None, max_episodes: int | None = None, status: str | None = None, min_score_count: int | None = None, w_cf: float = 0.55, w_content: float = 0.2, w_community: float = 0.25):
    """
    Collect CF, Content, and Community scores per user query and combine into hybrid score.
    Return list of highest scoring anime based on hybrid score DESC
        1. idk how to do the cf part actually
        2. import filters:build_mask, content:get_similarity. run get_similarity
        3. just get the score row from anime.csv
    """
    # Collaborative Filtering
    with open(f"{DATA_DIR}user_id_map.json", "r") as f:
        user_id_map = json.load(f)
    with open(f"{DATA_DIR}anime_id_map.json", "r") as f:
        anime_id_map = json.load(f)
    if user_id in user_id_map:  # No CF if no user data
        model = AnimeRecommender(max(user_id_map.values()) + 1, USER_EMBED_DIM, max(anime_id_map.values()) + 1, ANIME_EMBED_DIM)
        model.load_state_dict(torch.load(f"{ROOT_DIR}model.pth"))
        model.eval()

    pass