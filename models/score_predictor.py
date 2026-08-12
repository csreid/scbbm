import torch
from tqdm import tqdm
from torch.nn import (
	CrossEntropyLoss,
	Module,
	LSTM,
	Linear,
	Embedding,
	functional as F,
	RNN,
	MSELoss,
	BCEWithLogitsLoss,
	GRU,
)


class ScorePredictor(Module):
	def __init__(self, teams, embedding_dim=10):
		super().__init__()
		self.offense_embedding = Embedding(
			num_embeddings=len(teams), max_norm=1.0, embedding_dim=embedding_dim
		)
		self.defense_embedding = Embedding(
			num_embeddings=len(teams), max_norm=1.0, embedding_dim=embedding_dim
		)
		self.out = Linear(2 * embedding_dim, 2)

		self._teams = teams
		self._teams_idx_map = {t: idx for idx, t in enumerate(teams)}

		self.double()

	def forward(self, home_teams, away_teams):
		home_idx = torch.tensor([self._teams_idx_map[t] for t in home_teams])
		away_idx = torch.tensor([self._teams_idx_map[t] for t in away_teams])

		home_offense = self.offense_embedding(home_idx)
		home_defense = self.defense_embedding(home_idx)

		away_offense = self.offense_embedding(away_idx)
		away_defense = self.defense_embedding(away_idx)

		home_emb = home_offense - away_defense
		away_emb = away_offense - home_offense

		game_emb = torch.cat((home_emb, away_emb), dim=1)

		out = self.out(game_emb)

		return out
