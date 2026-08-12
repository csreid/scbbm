"""
Domain-specific game quality metrics for the basketball play-by-play model.

Metrics computed on autoregressive samples from the model:
  - team_membership_acc: fraction of predicted players who actually appear
    in the real game's roster (players seen in that team's real games)
  - rebound_after_miss_rate: fraction of missed shots followed by any rebound
  - sub_pairing_rate: fraction of sub_out events followed by sub_in within 3 plays
  - ft_adjacency_rate: fraction of free throws preceded by a shot/foul
  - home_win_rate: fraction of simulated games where home side makes more field goals
  - box_score_outlier_rate: fraction of per-player play counts that fall >N std devs
    outside the reference distribution

Call build_reference_stats() once at startup, then compute_metrics() + log_metrics()
every eval_every epochs.
"""

import collections
import random
import torch
from tqdm import tqdm

# Play types that are "misses" (should be followed by a rebound)
_MISS_TYPES = {"missed_2pt", "missed_3pt"}
_REBOUND_TYPES = {"offensive_rebound", "defensive_rebound", "team_rebound"}
_SHOT_TYPES = {"made_2pt", "missed_2pt", "made_3pt", "missed_3pt"}
_FT_TYPES = {"made_free_throw", "missed_free_throw"}
_FT_PRECURSORS = _SHOT_TYPES | {"foul"}


def build_reference_stats(dataset):
	"""
	Compute reference statistics from the real dataset.

	Returns a dict with:
	  ref['home_players']:       dict team_name -> set of player names seen on home side
	  ref['away_players']:       dict team_name -> set of player names seen on away side
	  ref['player_game_counts']: dict player_name -> list of per-game play counts
	  ref['rebound_after_miss']: float, empirical rate
	  ref['sub_pairing']:        float, empirical rate
	  ref['ft_adjacency']:       float, empirical rate

	Player side attribution uses the possession column baked into game_state.
	Specifically: plays in the first half are split by the sign of score_diff
	change. This is approximate; the real signal would require possession columns
	passed through from the raw CSV. In practice the per-game roster mask already
	guarantees membership; this is used to track *side* accuracy.
	"""
	# Resolve actual dataset (Subset wraps)
	base = dataset.dataset if hasattr(dataset, "dataset") else dataset
	indices = (
		dataset.indices if hasattr(dataset, "indices") else range(len(base))
	)

	play_types = base.play_types  # list: idx -> name
	players = base.players  # list: idx -> name
	team_labels = base._game_labels  # list: (home, away)

	# Map player -> which teams they appeared for as home / away
	# player -> {team -> count_home, team -> count_away}
	player_home_counts = collections.defaultdict(lambda: collections.Counter())
	player_away_counts = collections.defaultdict(lambda: collections.Counter())

	player_game_counts = collections.defaultdict(list)

	rebound_after_miss_num = 0
	rebound_after_miss_den = 0
	sub_pairing_num = 0
	sub_pairing_den = 0
	ft_adjacency_num = 0
	ft_adjacency_den = 0

	# Plays that typically belong to the scoring team
	_POSSESSION_PLAYS = _SHOT_TYPES | {"turnover", "offensive_rebound"}
	# Plays that typically belong to the non-scoring/defending team
	_DEFENSE_PLAYS = {"defensive_rebound", "steal", "block"}

	for real_idx in tqdm(indices, desc="Building reference stats", leave=False):
		teams_t, x_types, x_players, x_state, y_types, y_players, *_ = base[
			real_idx
		]
		home, away = team_labels[real_idx]

		all_types = [play_types[t.item()] for t in x_types] + [
			play_types[y_types[-1].item()]
		]
		all_players = [players[p.item()] for p in x_players] + [
			players[y_players[-1].item()]
		]
		# game_state[:, 0] is score_diff (home - away) / 30
		all_score_diff = (
			list(x_state[:, 0].tolist()) + [x_state[-1, 0].item()]
		)

		# Approximate side attribution via score_diff delta.
		# When score_diff increases, home team scored. When it decreases, away scored.
		for i, (t, p) in enumerate(zip(all_types, all_players)):
			if p in ("<PAD>", "<none>"):
				continue
			if t in _POSSESSION_PLAYS:
				# Scoring/turnover plays: use sign of score change at this step
				if i + 1 < len(all_score_diff):
					delta = all_score_diff[i + 1] - all_score_diff[i]
				else:
					delta = 0.0
				if delta > 0:
					player_home_counts[p][home] += 1
				elif delta < 0:
					player_away_counts[p][away] += 1
				# delta == 0 (turnover, non-scoring): ambiguous, skip
			elif t in _DEFENSE_PLAYS:
				# Defensive plays belong to whichever team DIDN'T have the ball;
				# we approximate that by flipping the scoring delta signal.
				if i + 1 < len(all_score_diff):
					delta = all_score_diff[i + 1] - all_score_diff[i]
				else:
					delta = 0.0
				if delta > 0:
					player_away_counts[p][away] += 1
				elif delta < 0:
					player_home_counts[p][home] += 1

		# Per-player play counts this game
		game_player_counts = collections.Counter(
			p for p in all_players if p not in ("<PAD>", "<none>")
		)
		for p, c in game_player_counts.items():
			player_game_counts[p].append(c)

		# Rebound-after-miss
		for i, t in enumerate(all_types[:-1]):
			if t in _MISS_TYPES:
				rebound_after_miss_den += 1
				if all_types[i + 1] in _REBOUND_TYPES:
					rebound_after_miss_num += 1

		# Sub pairing (sub_out → sub_in within 3 plays)
		for i, t in enumerate(all_types):
			if t == "sub_out":
				sub_pairing_den += 1
				window = all_types[i + 1 : i + 4]
				if "sub_in" in window:
					sub_pairing_num += 1

		# FT adjacency (free throw preceded by shot/foul/other FT)
		for i, t in enumerate(all_types):
			if t in _FT_TYPES:
				ft_adjacency_den += 1
				if i > 0 and all_types[i - 1] in _FT_PRECURSORS | _FT_TYPES:
					ft_adjacency_num += 1

	# Build canonical home/away team sets per player (plurality vote)
	player_home_teams = {}  # player -> set of teams they most often played home for
	player_away_teams = {}
	all_known_players = set(player_home_counts) | set(player_away_counts)
	for p in all_known_players:
		if player_home_counts[p]:
			player_home_teams[p] = set(player_home_counts[p].keys())
		if player_away_counts[p]:
			player_away_teams[p] = set(player_away_counts[p].keys())

	ref = {
		"player_home_teams": player_home_teams,
		"player_away_teams": player_away_teams,
		"player_game_counts": dict(player_game_counts),
		"rebound_after_miss": rebound_after_miss_num / max(1, rebound_after_miss_den),
		"sub_pairing": sub_pairing_num / max(1, sub_pairing_den),
		"ft_adjacency": ft_adjacency_num / max(1, ft_adjacency_den),
	}
	return ref


def simulate_n_games(model, dataset, n_games, n_steps=200, device="cpu"):
	"""
	Autoregressively simulate n_games games and return raw play sequences.

	Returns list of dicts, one per game:
	  {
	    'home': str,
	    'away': str,
	    'play_types':  list[str],
	    'players':     list[str],
	  }
	"""
	base = dataset.dataset if hasattr(dataset, "dataset") else dataset
	indices = (
		dataset.indices
		if hasattr(dataset, "indices")
		else list(range(len(base)))
	)

	chosen = random.sample(list(indices), min(n_games, len(indices)))
	results = []
	model.eval()

	for real_idx in tqdm(chosen, desc="Simulating games", leave=False):
		home, away, sim = model.simulate_game(
			dataset,
			list(indices).index(real_idx)
			if hasattr(dataset, "indices")
			else real_idx,
			n_steps=n_steps,
			device=device,
		)
		results.append(
			{
				"home": home,
				"away": away,
				"play_types": [s["pred_type"]   for s in sim],
				"players":    [s["pred_player"]  for s in sim],
			}
		)

	return results


def compute_metrics(simulated_games, ref, n_std=3.0):
	"""
	Compute domain-specific metrics over a list of simulated game dicts.

	Returns a flat dict of metric_name -> float, suitable for TensorBoard.
	"""
	if not simulated_games:
		return {}

	rebound_after_miss_num = 0
	rebound_after_miss_den = 0

	sub_pairing_num = 0
	sub_pairing_den = 0

	ft_adjacency_num = 0
	ft_adjacency_den = 0

	box_outlier_num = 0
	box_outlier_den = 0

	player_game_counts = ref["player_game_counts"]
	player_home_teams = ref.get("player_home_teams", {})
	player_away_teams = ref.get("player_away_teams", {})

	# player_side_acc: among plays with a named player, fraction where the
	# player has been seen playing for the home/away team in that capacity.
	player_side_hits = 0
	player_side_total = 0

	for game in simulated_games:
		home = game["home"]
		away = game["away"]
		pts = game["play_types"]
		pls = game["players"]

		# Player-side accuracy: check whether predicted player has been seen
		# playing for home or away in that role across training games.
		for p in pls:
			if p in ("<PAD>", "<none>"):
				continue
			p_home = player_home_teams.get(p, set())
			p_away = player_away_teams.get(p, set())
			if not p_home and not p_away:
				continue  # no side info available for this player
			player_side_total += 1
			if home in p_home or away in p_away:
				player_side_hits += 1

		# Rebound after miss
		for i, t in enumerate(pts[:-1]):
			if t in _MISS_TYPES:
				rebound_after_miss_den += 1
				if pts[i + 1] in _REBOUND_TYPES:
					rebound_after_miss_num += 1

		# Sub pairing
		for i, t in enumerate(pts):
			if t == "sub_out":
				sub_pairing_den += 1
				window = pts[i + 1 : i + 4]
				if "sub_in" in window:
					sub_pairing_num += 1

		# FT adjacency
		for i, t in enumerate(pts):
			if t in _FT_TYPES:
				ft_adjacency_den += 1
				if i > 0 and pts[i - 1] in _FT_PRECURSORS | _FT_TYPES:
					ft_adjacency_num += 1

		# Box score outliers: per-player play counts vs reference distribution
		sim_player_counts = collections.Counter(
			p for p in pls if p not in ("<PAD>", "<none>")
		)
		for p, count in sim_player_counts.items():
			hist = player_game_counts.get(p, [])
			if len(hist) < 5:
				continue  # not enough reference data
			import statistics

			mu = statistics.mean(hist)
			try:
				sigma = statistics.stdev(hist)
			except statistics.StatisticsError:
				continue
			if sigma < 0.5:
				sigma = 0.5  # avoid near-zero std
			box_outlier_den += 1
			if abs(count - mu) > n_std * sigma:
				box_outlier_num += 1

	metrics = {
		# Post-masking, team membership is structurally guaranteed; this measures
		# whether predicted players have been seen on the correct HOME or AWAY side.
		"metrics/player_side_acc": player_side_hits / max(1, player_side_total),
		"metrics/rebound_after_miss_rate": rebound_after_miss_num / max(1, rebound_after_miss_den),
		"metrics/sub_pairing_rate": sub_pairing_num / max(1, sub_pairing_den),
		"metrics/ft_adjacency_rate": ft_adjacency_num / max(1, ft_adjacency_den),
		"metrics/box_score_outlier_rate": box_outlier_num / max(1, box_outlier_den),
		# Reference baselines for quick comparison in the console log
		"metrics_ref/rebound_after_miss": ref["rebound_after_miss"],
		"metrics_ref/sub_pairing": ref["sub_pairing"],
		"metrics_ref/ft_adjacency": ref["ft_adjacency"],
	}
	return metrics


def log_metrics(metrics, writer, epoch):
	"""Write all metrics to TensorBoard."""
	for k, v in metrics.items():
		writer.add_scalar(k, v, epoch)
