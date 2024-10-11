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

class PlayNetwork(Module):
	def __init__(self, in_feats, n_plays):
		super().__init__()
		self.fc1 = Linear(in_feats, 2000)
		self.fc2 = Linear(in_feats, 1000)
		self.out = Linear(1000, n_plays)

	def forward(self, X):
		out = self.fc1(X)
		out = F.leaky_relu(out, negative_slope=0.1)
		out = self.fc2(X)
		out = F.leaky_relu(out, negative_slope=0.1)
		out = self.out(out)

		return out

class GameSimulator(Module):
	def __init__(self, n_teams, n_plays):
		super().__init__()
		self._n_plays = n_plays
		self._n_teams = n_teams

		team_feats = 10
		play_feats = 1000
		rec_feats = 1000
		layers = 1

		self.team_input = Embedding(n_teams, team_feats, max_norm=1.)
		self.play_input = Embedding(n_plays, play_feats, max_norm=1., padding_idx=0, scale_grad_by_freq=True)

		self.recurrent = LSTM(2 * team_feats + play_feats, rec_feats, num_layers=layers)
		self.play_out = PlayNetwork(rec_feats + n_teams, n_plays)
		self.device = 'cpu'
		#self.time_output = Linear(rec_feats, 1)

	def to(self, device):
		res = super().to(device)
		self.play_input = self.play_input.to('cpu')
		self.team_input = self.team_input.to('cpu')

		return res

	def forward(self, teams, plays, times, state=None):
		batch_n = plays.shape[1]
		seq_n = plays.shape[0]

		play_emb = self.play_input(plays)
		play_emb = play_emb.to('cuda:0')
		teams_emb = self.team_input(teams)
		teams_emb = teams_emb.to('cuda:0')

		flat_teams = torch.flatten(teams_emb, start_dim=2, end_dim=3)

		out = torch.cat((flat_teams, play_emb), dim=2)
		out, state = self.recurrent(out, state)

		teams_vec = torch.zeros(1, batch_n, self._n_teams)
		for t in teams:
			teams_vec[0, :, t] = 1.
		teams_vec = teams_vec.expand(seq_n, -1, -1).to(self.device)

		#play_out = self.play_out(out)
		out = torch.cat((teams_vec.to('cuda:0', non_blocking=True), out), dim=2)
		play_out = self.play_out(out)

		return play_out, state

if __name__ == '__main__':
	gs = GameSimulator(1, 1)
