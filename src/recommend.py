"""
Combine Collaborative Filtering, Content-Based Filtering, and Community Score into one recommendation score and rank scores
"""

import pandas as pd
import json
import torch

from dataclasses import dataclass
from config import ANIME_EMBED_DIM, DATA_DIR, ROOT_DIR, USER_EMBED_DIM
from model import AnimeRecommender
from data import load_anime
from content import genres_encode, type_encode, get_similarity
from filters import build_mask

@dataclass
class ContentQuery:
    genres: list[str] | None = None
    air_type: str | None = None

@dataclass
class FilterQuery:
    min_score: int | None = None
    max_episodes: int | None = None
    status: str | None = None
    min_score_count: int | None = None

@dataclass
class Weights:
    w_cf: float = 0.55
    w_content: float = 0.2
    w_community: float = 0.25

def recommend(user_id: str, query: ContentQuery = ContentQuery(), weights: Weights = Weights()):
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

        user_tensor = torch.tensor([user_id_map[user_id]] * len(anime_id_map), dtype=torch.long)
        anime_tensor = torch.arange(len(anime_id_map))

        with torch.no_grad():
            cf_preds = torch.clamp(model(user_tensor, anime_tensor), 1, 10) / 10.0
    else: cf_preds = None

    # Content-Based Filtering
    genre_query = pd.DataFrame(0, index=[0], columns=genres_encode.columns)
    type_query = pd.DataFrame(0, index=[0], columns=type_encode.columns)
    if query.genres:
        for genre in query.genres:
            genre_query[genre] = 1

    # TODO: redesign build_mask() to be used for all 3 rating types (cf, content, community)
    content_preds = get_similarity(genre_query, type_query, build_mask())