import torch
import torch.nn as nn

class AnimeRecommender(nn.Module):
    def __init__(self, num_users: int, embedding_dim_users: int, num_anime: int, embedding_dim_anime: int):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim_users)
        self.anime_embedding = nn.Embedding(num_anime, embedding_dim_anime)

    def forward(self, user_ids, anime_ids):
        user_embed = self.user_embedding(user_ids)
        anime_embed = self.anime_embedding(anime_ids)

        return (user_embed * anime_embed).sum(dim=1)