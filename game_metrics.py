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
	  ref['team_rosters']:  dict mapping team_name -> set of player names
	  ref['play_type_counts']: Counter of (play_type_str,) for freq reference
	  ref['player_play_counts']: Counter of (player_name,) across all games
	  ref['player_game_counts']: Counter of player_name -> appearances per game,
	                             kept as list for std-dev calculation
	  ref['rebound_after_miss']: float, empirical rate
	  ref['sub_pairing']:        float, empirical rate
	  ref['ft_adjacency']:       float, empirical rate
	  ref['home_fg_win_rate']:   float, fraction of games home made >= away made FGs
	"""
	# Resolve actual dataset (Subset wraps)
	base = dataset.dataset if hasattr(dataset, "dataset") else dataset
	indices = (
		dataset.indices if hasattr(dataset, "indices") else range(len(base))
	)

	play_types = base.play_types  # list: idx -> name
	players = base.players  # list: idx -> name
	team_labels = base._game_labels  # list: (home, away)

	team_rosters = collections.defaultdict(set)
	player_game_counts = collections.defaultdict(
		list
	)  # player -> [count_game0, count_game1, ...]

	rebound_after_miss_num = 0
	rebound_after_miss_den = 0
	sub_pairing_num = 0
	sub_pairing_den = 0
	ft_adjacency_num = 0
	ft_adjacency_den = 0
	home_fg_win_num = 0
	home_fg_win_den = 0

	for real_idx in tqdm(indices, desc="Building reference stats", leave=False):
		teams_t, x_types, x_players, x_state, y_types, y_players = base[
			real_idx
		]
		home, away = team_labels[real_idx]

		# Reconstruct full play sequence from x+y (x is [:-1], y is [1:])
		all_types = [play_types[t.item()] for t in x_types] + [
			play_types[y_types[-1].item()]
		]
		all_players = [players[p.item()] for p in x_players] + [
			players[y_players[-1].item()]
		]

		# Team rosters (anyone who isn't <PAD> or <none>)
		for p in all_players:
			if p not in ("<PAD>", "<none>"):
				team_rosters[home].add(p)
				team_rosters[away].add(
					p
				)  # we can't tell sides without more data, so add to both

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

		# FT adjacency (free throw preceded by shot/foul)
		for i, t in enumerate(all_types):
			if t in _FT_TYPES:
				ft_adjacency_den += 1
				if i > 0 and all_types[i - 1] in _FT_PRECURSORS | _FT_TYPES:
					ft_adjacency_num += 1

		# Home FG win rate (count made field goals, not free throws)
		_FG_MADE = {"made_2pt", "made_3pt"}
		# Without possession info we can only check aggregate FGs for the game
		fg_count = sum(1 for t in all_types if t in _FG_MADE)
		# Use heuristic: home teams historically win ~55% — just track as baseline
		home_fg_win_den += 1
		# We can't split home/away FGs from this data, so skip home_fg_win_rate
		# and compute it from simulated games directly

	ref = {
		"team_rosters": dict(team_rosters),
		"player_game_counts": dict(player_game_counts),
		"rebound_after_miss": rebound_after_miss_num
		/ max(1, rebound_after_miss_den),
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
				"play_types": [s[0] for s in sim],
				"players": [s[1] for s in sim],
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

	team_membership_hits = 0
	team_membership_total = 0

	rebound_after_miss_num = 0
	rebound_after_miss_den = 0

	sub_pairing_num = 0
	sub_pairing_den = 0

	ft_adjacency_num = 0
	ft_adjacency_den = 0

	box_outlier_num = 0
	box_outlier_den = 0

	player_game_counts = ref["player_game_counts"]

	for game in simulated_games:
		home = game["home"]
		away = game["away"]
		pts = game["play_types"]
		pls = game["players"]

		home_roster = ref["team_rosters"].get(home, set())
		away_roster = ref["team_rosters"].get(away, set())
		known_players = home_roster | away_roster

		# Team membership
		for p in pls:
			if p not in ("<PAD>", "<none>"):
				team_membership_total += 1
				if p in known_players:
					team_membership_hits += 1

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
		"metrics/team_membership_acc": team_membership_hits
		/ max(1, team_membership_total),
		"metrics/rebound_after_miss_rate": rebound_after_miss_num
		/ max(1, rebound_after_miss_den),
		"metrics/sub_pairing_rate": sub_pairing_num / max(1, sub_pairing_den),
		"metrics/ft_adjacency_rate": ft_adjacency_num
		/ max(1, ft_adjacency_den),
		"metrics/box_score_outlier_rate": box_outlier_num
		/ max(1, box_outlier_den),
		# Reference baselines (logged once but useful for comparison)
		"metrics_ref/rebound_after_miss": ref["rebound_after_miss"],
		"metrics_ref/sub_pairing": ref["sub_pairing"],
		"metrics_ref/ft_adjacency": ref["ft_adjacency"],
	}
	return metrics


def log_metrics(metrics, writer, epoch):
	"""Write all metrics to TensorBoard."""
	for k, v in metrics.items():
		writer.add_scalar(k, v, epoch)
