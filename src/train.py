"""
This script trains an AnimeRecommender model on the data saved by preprocess.py
Weights are saved to model.pth
"""

import torch
import torch.nn as nn
import pandas as pd

from config import ANIME_EMBED_DIM, EPOCHS, LEARNING_RATE, USER_EMBED_DIM, WEIGHT_DECAY, BATCH_SIZE
from data import load_processed_ratings
from model import AnimeRecommender
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ratings = load_processed_ratings()

user_ids = torch.tensor(ratings["user_id"].values, dtype=torch.long)
anime_ids = torch.tensor(ratings["anime_id"].values, dtype=torch.long)
scores = torch.tensor(ratings["score"].values, dtype=torch.float32)

user_train, user_test, anime_train, anime_test, scores_train, scores_test = \
    train_test_split(user_ids, anime_ids, scores, test_size=0.2, random_state=13)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, user_ids: torch.Tensor, anime_ids: torch.Tensor, scores: torch.Tensor):
        self.user_ids = user_ids
        self.anime_ids = anime_ids
        self.scores = scores
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, i):
        return [self.user_ids[i], self.anime_ids[i], self.scores[i]]

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(Dataset(user_train, anime_train, scores_train), BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset(user_test, anime_test, scores_test), BATCH_SIZE, shuffle=False)

    model = AnimeRecommender(num_users=int(user_ids.max().item() + 1), \
                            num_anime=int(anime_ids.max().item() + 1), \
                            embedding_dim_users=USER_EMBED_DIM, \
                            embedding_dim_anime=ANIME_EMBED_DIM).to(device)

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            predictions = model(batch[0].to(device), batch[1].to(device))
            mse = loss_fn(predictions, batch[2].to(device))
            train_loss += mse.item()

            optim.zero_grad()
            mse.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
        
        avg_train_loss = train_loss / len(train_loader)

        # Evaluate every epoch
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                predictions = model(batch[0].to(device), batch[1].to(device))
                mse = loss_fn(predictions, batch[2].to(device))
                test_loss += mse.item()
        
        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}", flush=True)

    torch.save(model.state_dict(), "model.pth")
