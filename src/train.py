from model import AnimeRecommender
import torch
import torch.nn as nn
import pandas as pd

DATA_DIR = "./data/"
BATCH_SIZE = 1024
USER_EMBED_DIM = 50
ANIME_EMBED_DIM = 50
LEARNING_RATE = 0.001
EPOCHS = 10

ratings = pd.read_csv(f"{DATA_DIR}ratings_sampled_mapped.csv")

user_ids = torch.tensor(ratings["user_id"].values, dtype=torch.long)
anime_ids = torch.tensor(ratings["anime_id"].values, dtype=torch.long)
scores = torch.tensor(ratings["score"].values, dtype=torch.float32)

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
    loader = torch.utils.data.DataLoader(Dataset(user_ids, anime_ids, scores), BATCH_SIZE, shuffle=True)

    model = AnimeRecommender(num_users=int(user_ids.max().item() + 1), \
                            num_anime=int(anime_ids.max().item() + 1), \
                            embedding_dim_users=USER_EMBED_DIM, \
                            embedding_dim_anime=ANIME_EMBED_DIM)

    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            predictions = model(batch[0], batch[1])
            mse = loss(predictions, batch[2])
            total_loss += mse.item()

            optim.zero_grad()
            mse.backward()
            optim.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "model.pth")