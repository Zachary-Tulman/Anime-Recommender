from model import AnimeRecommender
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    loader = torch.utils.data.DataLoader(Dataset(user_train, anime_train, scores_train), BATCH_SIZE, shuffle=True)

    model = AnimeRecommender(num_users=int(user_ids.max().item() + 1), \
                            num_anime=int(anime_ids.max().item() + 1), \
                            embedding_dim_users=USER_EMBED_DIM, \
                            embedding_dim_anime=ANIME_EMBED_DIM).to(device)

    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            predictions = model(batch[0].to(device), batch[1].to(device))
            mse = loss(predictions, batch[2].to(device))
            total_loss += mse.item()

            optim.zero_grad()
            mse.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(loader):.4f}", flush=True)

    torch.save(model.state_dict(), "model.pth")

    model.eval()
    with torch.no_grad():
        predictions = model(user_test.to(device), anime_test.to(device))
        mse = loss(predictions, scores_test.to(device))
    
    print(f"Test loss: {mse.item():.4f}")