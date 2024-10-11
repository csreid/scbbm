import torch
from tqdm import tqdm
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pandas as pd

from models.score_predictor import ScorePredictor

data = pd.read_csv('ncaahoopR_data/2023-24/pbp_logs/schedule.csv')
teams = set()
teams |= set(list(data['home'].values))
teams |= set(list(data['away'].values))
teams = sorted(list(teams))

class ScoreDataset(Dataset):
	def __init__(self, data):
		self._data = data

	def __getitem__(self, idx):
		home_team = self._data.iloc[idx].home
		away_team = self._data.iloc[idx].away
		home_score = self._data.iloc[idx].home_score
		away_score = self._data.iloc[idx].away_score

		return home_team, away_team, torch.tensor([home_score, away_score])

	def __len__(self):
		return len(self._data)

ds = ScoreDataset(data)
loader = DataLoader(ds, batch_size=64)
model = ScorePredictor(teams, embedding_dim=4)
loss_fn = MSELoss()
opt = Adam(model.parameters())

for epoch in tqdm(range(200)):
	for batch in loader:
		home, away, true_scores = next(iter(loader))

		pred_scores = model(home, away)

		loss = loss_fn(pred_scores, true_scores)

		opt.zero_grad()
		loss.backward()
		opt.step()

		tqdm.write(f'{loss.item()}')

torch.save(model.state_dict(), 'score_model.pt')
