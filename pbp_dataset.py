import os
import re
import glob
import difflib
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


# Fixed play type vocabulary. Index 0 is <PAD> and is ignored in loss.
PLAY_TYPES = [
	"<PAD>",
	"made_2pt",
	"missed_2pt",
	"made_3pt",
	"missed_3pt",
	"made_free_throw",
	"missed_free_throw",
	"offensive_rebound",
	"defensive_rebound",
	"team_rebound",
	"turnover",
	"steal",
	"block",
	"foul",
	"jump_ball",
	"sub_in",
	"sub_out",
	"end_of_period",
	"other",
]

PLAY_TYPE_TO_IDX = {t: i for i, t in enumerate(PLAY_TYPES)}

# Known team name aliases: alternate spelling -> canonical name.
# Run find_duplicate_teams() on your dataset to discover new ones.
TEAM_ALIASES = {
	"Purdue Fort Wayn": "Purdue Fort Wayne",
	"Purdue FW": "Purdue Fort Wayne",
	"PFW": "Purdue Fort Wayne",
	"Michigan St": "Michigan State",
	"Mississippi St": "Mississippi State",
	"Ohio St": "Ohio State",
	"Penn St": "Penn State",
	"Florida St": "Florida State",
	"Iowa St": "Iowa State",
	"Kansas St": "Kansas State",
	"Oklahoma St": "Oklahoma State",
	"Oregon St": "Oregon State",
	"Arizona St": "Arizona State",
	"Colorado St": "Colorado State",
	"Utah St": "Utah State",
	"Washington St": "Washington State",
	"Wichita St": "Wichita State",
	"Boise St": "Boise State",
	"San Diego St": "San Diego State",
	"Fresno St": "Fresno State",
	"Sacramento St": "Sacramento State",
	"Cal St Fullerton": "Cal State Fullerton",
	"Cal St Bakersfield": "Cal State Bakersfield",
	"Cal St Northridge": "Cal State Northridge",
	"UCSB": "UC Santa Barbara",
	"LIU": "Long Island University",
	"SIU Edwardsville": "SIU-Edwardsville",
	"SIUE": "SIU-Edwardsville",
}


def normalize_team_name(name):
	"""Map known alternate team name spellings to their canonical form."""
	return TEAM_ALIASES.get(name, name)


def find_duplicate_teams(df_or_path, cutoff=0.85):
	"""
	Print likely duplicate team name pairs using fuzzy string matching.
	Run this on your dataset to discover aliases to add to TEAM_ALIASES.
	"""
	if isinstance(df_or_path, str):
		df = (
			pd.read_parquet(df_or_path)
			if df_or_path.endswith(".parquet")
			else pd.read_csv(df_or_path)
		)
	else:
		df = df_or_path
	teams = sorted(set(df["home"].dropna()) | set(df["away"].dropna()))
	found = 0
	for i, t in enumerate(teams):
		matches = difflib.get_close_matches(
			t, teams[i + 1 :], n=3, cutoff=cutoff
		)
		for m in matches:
			print(f"  {repr(t):45s}  ~  {repr(m)}")
			found += 1
	print(f"{found} potential duplicates found (cutoff={cutoff})")


def _parse_play(row):
	"""
	Extract (play_type, ego_player) from a raw hoopR PBP row.

	Returns a (str, str) tuple where ego_player is '<none>' when there is no
	meaningful individual player associated with the event.
	"""
	desc = str(row.get("description", ""))

	# --- Shot events: use pre-parsed structured columns ---
	if pd.notna(row.get("shot_outcome")):
		made = row["shot_outcome"] == "made"
		if row.get("free_throw") == True:
			play_type = "made_free_throw" if made else "missed_free_throw"
		elif row.get("three_pt") == True:
			play_type = "made_3pt" if made else "missed_3pt"
		else:
			play_type = "made_2pt" if made else "missed_2pt"
		shooter = row.get("shooter")
		return play_type, shooter if pd.notna(shooter) else "<none>"

	# --- Foul events ---
	if row.get("foul") == True:
		m = re.match(r"Foul on (.+?)\.", desc)
		return "foul", m.group(1) if m else "<none>"

	# --- Substitutions ---
	m = re.match(r"(.+?) subbing in for", desc)
	if m:
		return "sub_in", m.group(1)

	m = re.match(r"(.+?) subbing out for", desc)
	if m:
		return "sub_out", m.group(1)

	# --- Team rebounds (no individual player) ---
	if "Team Rebound" in desc or "Deadball" in desc:
		return "team_rebound", "<none>"

	# --- Individual rebounds ---
	m = re.match(r"(.+?) Offensive Rebound", desc)
	if m:
		return "offensive_rebound", m.group(1)

	m = re.match(r"(.+?) Defensive Rebound", desc)
	if m:
		return "defensive_rebound", m.group(1)

	# --- Turnovers / steals / blocks ---
	m = re.match(r"(.+?) Turnover", desc)
	if m:
		return "turnover", m.group(1)

	m = re.match(r"(.+?) Steal", desc)
	if m:
		return "steal", m.group(1)

	m = re.match(r"(.+?) Block", desc)
	if m:
		return "block", m.group(1)

	# --- Administrative ---
	if "Jump Ball" in desc:
		return "jump_ball", "<none>"

	if desc.startswith("End of") or desc.startswith("Start"):
		return "end_of_period", "<none>"

	return "other", "<none>"


class PlayByPlayDataset(Dataset):
	"""
	Expects a CSV or parquet in the raw hoopR PBP format (as in sample_data.csv).
	For each game, __getitem__ returns an (X, Y) pair of consecutive play
	sequences suitable for next-play prediction.

	Player vocab: index 0 = <PAD> (ignored in loss), index 1 = <none>
	Play type vocab: index 0 = <PAD> (ignored in loss)

	Game state features (3,): [score_diff / 30, secs_remaining / 2400, half / 2]
	All three are roughly in [-1, 1] / [0, 1] range for typical games.
	Overtime halves will have half > 1; that's a useful signal, not a bug.

	Pass cache_path to save/load a parsed parquet so the slow parse step only
	runs once across training runs.
	"""

	def __init__(self, df_path, cache_path=None):
		if cache_path and os.path.exists(cache_path):
			print(f"Loading cache from {cache_path}...")
			df = pd.read_parquet(cache_path)
		else:
			print(f"Loading {df_path}...")
			df = (
				pd.read_parquet(df_path)
				if df_path.endswith(".parquet")
				else pd.read_csv(df_path)
			)

			print("Parsing play types and players...")
			parsed = df.apply(_parse_play, axis=1, result_type="expand")
			df["play_type"] = parsed[0]
			df["ego_player"] = parsed[1]

			if cache_path:
				print(f"Saving cache to {cache_path}...")
				df.to_parquet(cache_path, index=False)

		# Team vocab — normalize aliases before building the index
		all_teams = sorted(
			set(normalize_team_name(t) for t in df["home"].dropna())
			| set(normalize_team_name(t) for t in df["away"].dropna())
		)
		self.teams = all_teams
		self.team_to_idx = {t: i for i, t in enumerate(all_teams)}

		# Play type vocab is fixed (see PLAY_TYPES above)
		self.play_types = PLAY_TYPES
		self.play_type_to_idx = PLAY_TYPE_TO_IDX

		# Player vocab: 0=<PAD>, 1=<none>, 2+=actual players
		all_players = sorted(
			p for p in df["ego_player"].unique() if p != "<none>"
		)
		self.players = ["<PAD>", "<none>"] + all_players
		self.player_to_idx = {p: i for i, p in enumerate(self.players)}

		# Pre-convert every game to tensors so __getitem__ is a free list lookup.
		print("  Pre-processing games...")
		none_idx = self.player_to_idx["<none>"]
		self._games = []
		self._game_labels = []        # (home, away) strings for display
		self._game_player_masks = []  # per-game boolean roster masks
		for _, game in tqdm(
			df.groupby("game_id", sort=False), total=df["game_id"].nunique()
		):
			home = normalize_team_name(game.iloc[0]["home"])
			away = normalize_team_name(game.iloc[0]["away"])
			teams = torch.tensor(
				[self.team_to_idx[home], self.team_to_idx[away]]
			)

			play_types = torch.tensor(
				[
					self.play_type_to_idx.get(pt, PLAY_TYPE_TO_IDX["other"])
					for pt in game["play_type"]
				],
				dtype=torch.long,
			)

			players = torch.tensor(
				[
					self.player_to_idx.get(p, none_idx)
					for p in game["ego_player"]
				],
				dtype=torch.long,
			)

			raw_score_diff = torch.tensor(
				game["score_diff"].fillna(0).values, dtype=torch.float32
			)
			raw_secs = torch.tensor(
				game["secs_remaining"].fillna(0).values, dtype=torch.float32
			)
			half = (
				torch.tensor(game["half"].fillna(1).values, dtype=torch.float32)
				/ 2.0
			)

			# Time elapsed FOR each play: secs_remaining[i-1] - secs_remaining[i].
			# First play gets 0 (no prior play). Half transitions (where secs jump
			# upward) produce negative diffs — clamp those to 0.
			time_elapsed = torch.cat([
				torch.zeros(1),
				(raw_secs[:-1] - raw_secs[1:]).clamp(min=0),
			]).clamp(max=120)   # cap at 2 minutes

			# Score change AT each play: score_diff[i] - score_diff[i-1].
			# First play gets 0. Clip to ±3 (max single-play change) and shift
			# to class index 0–6 so CrossEntropyLoss can consume it directly.
			score_delta_raw = torch.cat([
				torch.zeros(1),
				raw_score_diff[1:] - raw_score_diff[:-1],
			])
			score_delta_cls = score_delta_raw.round().long().clamp(-3, 3) + 3

			# game_state input features (3,):
			#   [score_diff/30,  log1p(time_elapsed)/log1p(30),  half/2]
			# log1p(30) ≈ shot-clock length → most plays land near [0, 1].
			score_diff_norm = raw_score_diff / 30.0
			time_delta_norm = torch.log1p(time_elapsed) / torch.log1p(torch.tensor(30.0))
			game_state = torch.stack([score_diff_norm, time_delta_norm, half], dim=1)

			# Regression target in log1p-seconds space (un-normalised), used for
			# MSE loss so the model learns absolute time scale, not relative.
			time_delta_log = torch.log1p(time_elapsed)

			self._games.append((teams, play_types, players, game_state, score_delta_cls, time_delta_log))
			self._game_labels.append((home, away))

			# Per-game roster mask: True = this player may appear in this game.
			# Index 0 (PAD) is always False; index 1 (<none>) is always True.
			# Used to restrict the player head to the ~20 in-game players.
			game_player_set = set(players.tolist())
			game_player_set.discard(0)   # PAD is never a valid prediction
			game_player_set.add(1)       # <none> is always valid
			mask = torch.zeros(len(self.players), dtype=torch.bool)
			mask[list(game_player_set)] = True
			self._game_player_masks.append(mask)

		print(
			f"  {len(self._games)} games, {len(self.teams)} teams, {len(self.players)} players"
		)

	def __len__(self):
		return len(self._games)

	def __getitem__(self, idx):
		teams, play_types, players, game_state, score_delta_cls, time_delta_log = self._games[idx]
		mask = self._game_player_masks[idx]
		return (
			teams,
			play_types[:-1],
			players[:-1],
			game_state[:-1],
			play_types[1:],
			players[1:],
			mask,
			score_delta_cls[1:],   # score change that happens AT the next play
			time_delta_log[1:],    # log1p(secs) elapsed for the next play
		)


def train_val_split(dataset, val_fraction=0.1, seed=42):
	"""
	Split a PlayByPlayDataset into (train, val) Subset objects.
	The split is at game level, stratified by nothing — just a random shuffle.
	"""
	import random

	rng = random.Random(seed)
	indices = list(range(len(dataset)))
	rng.shuffle(indices)
	n_val = max(1, int(len(indices) * val_fraction))
	return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])


def build_dataframe(data_dir, seasons=None):
	"""
	Concatenate raw hoopR PBP CSVs from data_dir.
	seasons: list of season strings like ['2022-23', '2023-24'], or None for all.
	"""
	season_list = (
		seasons
		if seasons
		else [
			d
			for d in os.listdir(data_dir)
			if os.path.isdir(os.path.join(data_dir, d)) and d[0].isdigit()
		]
	)
	files = []
	for season in season_list:
		pattern = os.path.join(data_dir, season, "pbp_logs", "**", "*.csv")
		files.extend(glob.glob(pattern, recursive=True))
	if not files:
		raise FileNotFoundError(
			f"No PBP files found in {data_dir} for seasons {season_list}"
		)
	print(
		f"Found {len(files)} game files across {len(season_list)} season(s)..."
	)
	dfs = [pd.read_csv(f) for f in tqdm(files, desc="Reading CSVs")]
	return pd.concat(dfs, ignore_index=True)


def collate_fn(batch):
	"""Pad variable-length game sequences to the longest in the batch."""
	teams, x_types, x_players, x_state, y_types, y_players, player_masks, y_score_delta, y_time_delta_log = zip(*batch)

	return (
		torch.stack(teams),              # (batch, 2)
		pad_sequence(x_types),           # (seq, batch)
		pad_sequence(x_players),         # (seq, batch)
		pad_sequence(x_state),           # (seq, batch, 3)
		pad_sequence(y_types),           # (seq, batch)
		pad_sequence(y_players),         # (seq, batch)
		torch.stack(player_masks),       # (batch, n_players)
		pad_sequence(y_score_delta),     # (seq, batch)  long, class 0-6
		pad_sequence(y_time_delta_log),  # (seq, batch)  float, log1p seconds
	)
