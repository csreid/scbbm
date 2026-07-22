import torch
from torch.nn import (
	Module,
	LSTM,
	Linear,
	Embedding,
	functional as F,
)

from pbp_dataset import PLAY_TYPE_TO_IDX

_END_OF_PERIOD_IDX = PLAY_TYPE_TO_IDX["end_of_period"]
_LOG1P_30 = torch.log1p(torch.tensor(30.0))


class GameSimulator(Module):
	def __init__(self, n_teams, n_play_types, n_players):
		super().__init__()

		team_feats = 16
		play_type_feats = 8
		player_feats = 16
		game_state_feats = 3  # score_diff/30, log1p(time_delta)/log1p(30), half/2
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

		# All output heads beyond play_type are conditioned on the *next* play
		# type embedding (rec_feats + play_type_feats input).
		cond_feats = rec_feats + play_type_feats

		self.play_type_head = Linear(rec_feats, n_play_types)
		self.player_head = Linear(cond_feats, n_players)
		# 7-class: score change ∈ {-3,-2,-1,0,1,2,3} → class index = delta+3
		self.score_delta_head = Linear(cond_feats, 7)
		# Regression: predict log1p(seconds elapsed) for the next play
		self.time_delta_head = Linear(cond_feats, 1)

	def forward(
		self,
		teams,
		play_types,
		players,
		game_state,
		target_play_types=None,
		state=None,
		player_mask=None,
	):
		"""
		teams:             (batch, 2)          home/away team indices
		play_types:        (seq, batch)        input play type indices
		players:           (seq, batch)        input ego player indices (0=PAD, 1=<none>)
		game_state:        (seq, batch, 3)     [score_diff/30, log1p(td)/log1p(30), half/2]
		target_play_types: (seq, batch)        next play type for conditioning heads;
		                                       use y_types during training (teacher
		                                       forcing), omit at inference.
		state:             optional LSTM state
		player_mask:       (batch, n_players)  boolean, True = valid for this game.

		returns:
		  play_type_logits:   (seq, batch, n_play_types)
		  player_logits:      (seq, batch, n_players)
		  score_delta_logits: (seq, batch, 7)
		  time_delta_pred:    (seq, batch, 1)   log1p(seconds)
		  state:              LSTM state
		"""
		seq_n, batch_n = play_types.shape

		teams_emb = self.team_input(teams)           # (batch, 2, team_feats)
		flat_teams = teams_emb.view(batch_n, -1)     # (batch, 2*team_feats)
		flat_teams = flat_teams.unsqueeze(0).expand(seq_n, -1, -1)

		play_type_emb = self.play_type_input(play_types)   # (seq, batch, pt_feats)
		player_emb    = self.player_input(players)          # (seq, batch, pl_feats)

		lstm_in = torch.cat([flat_teams, play_type_emb, player_emb, game_state], dim=2)
		hidden, state = self.recurrent(lstm_in, state)     # (seq, batch, rec_feats)

		play_type_logits = self.play_type_head(hidden)     # (seq, batch, n_play_types)

		# Condition all remaining heads on the *next* play type.
		cond_emb = self.play_type_input(
			target_play_types if target_play_types is not None else play_types
		)
		cond = torch.cat([hidden, cond_emb], dim=2)        # (seq, batch, cond_feats)

		player_logits      = self.player_head(cond)        # (seq, batch, n_players)
		score_delta_logits = self.score_delta_head(cond)   # (seq, batch, 7)
		time_delta_pred    = self.time_delta_head(cond)    # (seq, batch, 1)

		if player_mask is not None:
			player_logits = player_logits.masked_fill(
				~player_mask.unsqueeze(0), float("-inf")
			)

		return play_type_logits, player_logits, score_delta_logits, time_delta_pred, state

	@torch.no_grad()
	def simulate_game(self, dataset, game_idx, n_steps=30, device="cpu"):
		"""
		Fully autoregressive simulation: the model generates its own game state
		(score diff, time elapsed, half) from its predictions rather than copying
		from the real game. Seeded by the real first play's state.

		Returns:
		  home, away  — team name strings
		  list of dicts with keys:
		    pred_type, pred_player, pred_score_delta, pred_time_delta,
		    true_type, true_player
		"""
		self.eval()
		base = dataset.dataset if hasattr(dataset, "dataset") else dataset
		real_idx = dataset.indices[game_idx] if hasattr(dataset, "indices") else game_idx

		teams, x_types, x_players, x_state, y_types, y_players, player_mask, *_ = dataset[game_idx]
		home, away = base._game_labels[real_idx]
		roster_mask = player_mask.to(device)

		n_steps = min(n_steps, len(x_types))
		teams_t = teams.unsqueeze(0).to(device)  # (1, 2)

		# Seed: use real first play's state
		cur_type   = x_types[0:1].unsqueeze(1).to(device)    # (1, 1)
		cur_player = x_players[0:1].unsqueeze(1).to(device)  # (1, 1)
		cur_state  = x_state[0].unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3)

		# Tracked game state for full autoregression
		score_diff = x_state[0, 0].item() * 30.0  # un-normalise
		half = max(1, round(x_state[0, 2].item() * 2.0))

		results = []
		state = None
		log1p_30 = _LOG1P_30.to(device)

		for step in range(n_steps):
			# One LSTM step
			teams_emb = self.team_input(teams_t)
			flat_teams = teams_emb.view(1, -1).unsqueeze(0)
			lstm_in = torch.cat([
				flat_teams,
				self.play_type_input(cur_type),
				self.player_input(cur_player),
				cur_state,
			], dim=2)
			hidden, state = self.recurrent(lstm_in, state)

			# --- Play type ---
			type_probs = F.softmax(self.play_type_head(hidden)[0, 0, 1:], dim=0)
			next_type  = torch.multinomial(type_probs, 1).item() + 1

			# --- Condition embedding for remaining heads ---
			pt_emb = self.play_type_input(torch.tensor([[next_type]], device=device))
			cond   = torch.cat([hidden, pt_emb], dim=2)

			# --- Player ---
			raw_p = self.player_head(cond)[0, 0].clone()
			raw_p[~roster_mask] = float("-inf")
			next_player = torch.multinomial(F.softmax(raw_p[1:], dim=0), 1).item() + 1

			# --- Score delta ---
			sd_probs    = F.softmax(self.score_delta_head(cond)[0, 0], dim=0)
			pred_delta_cls = torch.multinomial(sd_probs, 1).item()
			pred_delta  = pred_delta_cls - 3  # back to ±3 range
			score_diff += pred_delta

			# --- Time delta ---
			pred_log_td = self.time_delta_head(cond)[0, 0, 0].item()
			pred_secs   = torch.expm1(torch.tensor(pred_log_td)).clamp(min=0).item()

			# --- Advance half ---
			if next_type == _END_OF_PERIOD_IDX:
				half += 1

			# --- Build next input game_state ---
			td_norm = torch.log1p(torch.tensor(pred_secs, device=device)) / log1p_30
			cur_state = torch.tensor([[
				score_diff / 30.0,
				td_norm.item(),
				half / 2.0,
			]], device=device).unsqueeze(0)  # (1, 1, 3)

			pt_vocab = base.play_types
			pl_vocab = base.players
			results.append({
				"pred_type":        pt_vocab[next_type],
				"pred_player":      pl_vocab[next_player],
				"pred_score_delta": pred_delta,
				"pred_time_delta":  round(pred_secs, 1),
				"true_type":        pt_vocab[y_types[step].item()],
				"true_player":      pl_vocab[y_players[step].item()],
			})

			cur_type   = torch.tensor([[next_type]],   device=device)
			cur_player = torch.tensor([[next_player]], device=device)

		return home, away, results
