import torch
from torch.nn import CrossEntropyLoss, Module, LSTM, Linear, Embedding, functional as F, RNN, MSELoss, BCEWithLogitsLoss

class GameSimulator(Module):
	def __init__(self, n_teams, n_plays):
		super().__init__()
		self._n_plays = n_plays
		self._n_teams = n_teams

		team_feats = 10
		play_feats = 200

		self.team_input = Embedding(n_teams, team_feats, max_norm=1.)
		self.play_input = Embedding(n_plays, play_feats, max_norm=1., padding_idx=0)

		self.recurrent_score = LSTM(team_feats, 500, num_layers=1)
		self.score_out = Linear(500, 2)
		self.score_time_out = Linear(500, 1)

		self.final_score_out = Linear(team_feats, 2)

		self.recurrent1 = LSTM(2 * team_feats + play_feats + 1, 500, num_layers=2)
		self.recurrent2 = LSTM(500, 500, num_layers=1)
		self.play_fc = Linear(500, 500)
		self.play_out = Linear(500, n_plays)
		self.time_output = Linear(500, 1)

		self.team_filter = Linear(n_teams, n_plays)
		torch.nn.init.zeros_(self.team_filter.weight)

	def forward(self, teams, plays, times):
		batch_n = plays.shape[1]
		seq_n = plays.shape[0]

		play_emb = self.play_input(plays)

		with torch.no_grad():
			flat_teams = torch.flatten(self.team_input(teams), start_dim=1, end_dim=2)

		teams_emb = (flat_teams).unsqueeze(0).expand(seq_n, -1, -1)

		out = torch.cat((teams_emb, play_emb, times), dim=2)
		out, _ = self.recurrent1(out)

		out1 = self.play_fc(out)
		out1 = self.play_out(out)

		out2, _ = self.recurrent2(out)
		out2  = self.time_output(out2)

		dev = next(self.parameters()).device
		team_filter_input = torch.zeros(len(teams), self._n_teams)
		for (h_idx, a_idx), row in zip(teams, team_filter_input):
			row[h_idx] = 1.
			row[a_idx] = 1.

		team_filter_input = team_filter_input.expand(1, len(teams), self._n_teams).to(dev)
		filter_result = self.team_filter(team_filter_input)

		out1 = out1 * filter_result

		return out1, out2

	def scores(self, teams, length=200):
		with torch.no_grad():
			teams = self.team_input(teams)
		teams = torch.diff(teams, dim=1)
		teams = teams.squeeze(1)

		batch_size, team_feats = teams.shape

		teams = teams.expand(length, batch_size, team_feats)

		out, _ = self.recurrent_score(teams)
		out = F.sigmoid(out)

		score_out = F.sigmoid(self.score_out(out)) * 200
		time_out = F.sigmoid(self.score_time_out(out)) * 3600.

		return score_out, time_out

	def final_score(self, teams):
		out = self.team_input(teams)
		out = torch.diff(out, dim=1)
		out = out.squeeze(1)
		out = self.final_score_out(out)

		return out

if __name__ == '__main__':
	gs = GameSimulator(1, 1)
