import torch
from torch.nn import (
	Module,
	LSTM,
	Linear,
	Embedding,
	functional as F,
)

from pbp_dataset import PLAY_TYPE_TO_IDX


class GameSimulator(Module):
	def __init__(self, n_teams, n_play_types, n_players):
		super().__init__()

		team_feats = 32
		play_type_feats = 32
		player_feats = 128
		game_state_feats = 3  # score_diff, secs_remaining, half
		rec_feats = 512

		# index 0 is <PAD> for all vocabs; index 1 is <none> for players
		self.team_input = Embedding(n_teams, team_feats, max_norm=1.0)
		self.play_type_input = Embedding(
			n_play_types, play_type_feats, padding_idx=0
		)
		self.player_input = Embedding(n_players, player_feats, padding_idx=0)

		lstm_in_feats = (
			2 * team_feats + play_type_feats + player_feats + game_state_feats
		)
		self.recurrent = LSTM(
			lstm_in_feats, rec_feats, num_layers=2, dropout=0.1
		)

		self.play_type_head = Linear(rec_feats, n_play_types)
		# Player head is conditioned on hidden state + the *next* play type embedding.
		# Pass target_play_types during training (teacher forcing) and the predicted
		# play type during inference — see simulate_game().
		self.player_head = Linear(rec_feats + play_type_feats, n_players)

	def forward(
		self,
		teams,
		play_types,
		players,
		game_state,
		target_play_types=None,
		state=None,
	):
		"""
		teams:             (batch, 2)       home/away team indices
		play_types:        (seq, batch)     input play type indices
		players:           (seq, batch)     input ego player indices (0=PAD, 1=<none>)
		game_state:        (seq, batch, 3)  [score_diff, secs_remaining, half]
		target_play_types: (seq, batch)     next play type indices for player head
		                                    conditioning; use y_types during training,
		                                    omit (falls back to play_types) otherwise.
		state:             optional LSTM state

		returns:
		  play_type_logits: (seq, batch, n_play_types)
		  player_logits:    (seq, batch, n_players)
		  state:            LSTM state
		"""
		seq_n, batch_n = play_types.shape

		teams_emb = self.team_input(teams)  # (batch, 2, team_feats)
		flat_teams = teams_emb.view(batch_n, -1)  # (batch, 2*team_feats)
		flat_teams = flat_teams.unsqueeze(0).expand(
			seq_n, -1, -1
		)  # (seq, batch, 2*team_feats)

		play_type_emb = self.play_type_input(
			play_types
		)  # (seq, batch, play_type_feats)
		player_emb = self.player_input(players)  # (seq, batch, player_feats)

		lstm_in = torch.cat(
			[flat_teams, play_type_emb, player_emb, game_state], dim=2
		)
		hidden, state = self.recurrent(
			lstm_in, state
		)  # (seq, batch, rec_feats)

		play_type_logits = self.play_type_head(
			hidden
		)  # (seq, batch, n_play_types)

		# Condition player head on the *target* play type when available (training),
		# otherwise fall back to the input play type (shifted by one, acceptable at inference).
		cond_emb = self.play_type_input(
			target_play_types if target_play_types is not None else play_types
		)
		player_logits = self.player_head(torch.cat([hidden, cond_emb], dim=2))

		return play_type_logits, player_logits, state

	@torch.no_grad()
	def simulate_game(self, dataset, game_idx, n_steps=30, device="cpu"):
		"""
		Autoregressively generate n_steps plays for a game in dataset.

		Uses actual game_state features from the chosen game so timing and
		score context are realistic, but play type and player are sampled
		from the model's own predictions at each step.

		Returns list of (pred_play_type, pred_player, true_play_type, true_player).
		"""
		self.eval()
		teams, x_types, x_players, x_state, y_types, y_players = dataset[
			game_idx
		]

		# Unwrap Subset if needed
		if hasattr(dataset, "dataset"):
			home, away = dataset.dataset._game_labels[dataset.indices[game_idx]]
		else:
			home, away = dataset._game_labels[game_idx]

		n_steps = min(n_steps, len(x_types))
		teams_t = teams.unsqueeze(0).to(device)  # (1, 2)

		# Seed with the actual first play
		cur_type = x_types[0:1].unsqueeze(1).to(device)  # (1, 1)
		cur_player = x_players[0:1].unsqueeze(1).to(device)  # (1, 1)

		results = []
		state = None

		for step in range(n_steps):
			gs = x_state[step].unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3)

			# Run one LSTM step to get hidden state
			play_type_emb = self.play_type_input(cur_type)
			player_emb = self.player_input(cur_player)
			teams_emb = self.team_input(teams_t)
			flat_teams = teams_emb.view(1, -1).unsqueeze(0)
			lstm_in = torch.cat(
				[flat_teams, play_type_emb, player_emb, gs], dim=2
			)
			hidden, state = self.recurrent(lstm_in, state)

			# Predict next play type (skip PAD=0)
			type_logits = self.play_type_head(hidden)
			type_probs = F.softmax(type_logits[0, 0, 1:], dim=0)
			next_type = torch.multinomial(type_probs, 1).item() + 1

			# Predict player conditioned on the predicted play type
			pred_type_emb = self.play_type_input(
				torch.tensor([[next_type]], device=device)
			)
			player_logits = self.player_head(
				torch.cat([hidden, pred_type_emb], dim=2)
			)
			player_probs = F.softmax(player_logits[0, 0, 1:], dim=0)
			next_player = torch.multinomial(player_probs, 1).item() + 1

			results.append(
				(
					dataset.dataset.play_types[next_type]
					if hasattr(dataset, "dataset")
					else dataset.play_types[next_type],
					dataset.dataset.players[next_player]
					if hasattr(dataset, "dataset")
					else dataset.players[next_player],
					dataset.dataset.play_types[y_types[step].item()]
					if hasattr(dataset, "dataset")
					else dataset.play_types[y_types[step].item()],
					dataset.dataset.players[y_players[step].item()]
					if hasattr(dataset, "dataset")
					else dataset.players[y_players[step].item()],
				)
			)

			cur_type = torch.tensor([[next_type]], device=device)
			cur_player = torch.tensor([[next_player]], device=device)

		return home, away, results
