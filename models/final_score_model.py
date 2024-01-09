import torch
from torch.nn import (
	CrossEntropyLoss,
	Module,
	LSTM,
	Linear,
	Embedding,
	functional as F,
	RNN,
	MSELoss,
	BCEWithLogitsLoss
)

class FinalScoreModel(Module):
	def __init__(self, n_teams):
		super().__init__()

		self.off_embedding = Embedding(n_teams, embedding_dim=10)
		self.def_embedding = Embedding(n_teams, embedding_dim=10)

		self.fc1 = Linear(10, 16)
		self.out = Linear(16, 2)

	def forward(self, X, ds=None):
		if ds is not None:
			home = ds._team_id_map[X[0]]
			away = ds._team_id_map[X[1]]

			X = torch.tensor([[home, away]])

		off_s = self.off_embedding(X)
		def_s = self.def_embedding(X)

		emb_s = off_s[:, 0, :] - def_s[:, 1, :]
		emb_s = emb_s.squeeze()

		out = self.fc1(emb_s)
		out = F.tanh(out)
		out = self.out(out)

		return out
