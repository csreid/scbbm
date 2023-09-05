import pandas as pd
import torch
import numpy as np
import math
from torch.utils.data import Dataset

class PlayByPlayDataset(Dataset):
	def __init__(self, df_path, min_len, max_len):
		self.df = pd.read_csv('cleaned_data.csv')
		homes = self.df['home'].unique()
		aways = self.df['away'].unique()
		self.teams = list(set(homes).union(set(aways)))
		self._n_games = self.df['game_id'].nunique()

		self.teams_id_map = dict([(val, idx) for idx, val in enumerate(self.teams)])
		self.game_id_map = (
			dict(
				enumerate(
					list(self.df['game_id'].unique())
				)
			)
		)

		ranges = []
		new_rows = []
		for idx, game_id in enumerate(self.df.game_id.unique()):
			game = self.df[self.df.game_id == game_id]
			game_idx = game.index

			min_idx = game_idx[0]
			max_idx = game_idx[-1]
			home = game.iloc[0].home
			away = game.iloc[0].away

			new_rows.append({
				"game_id": game_id,
				"home": home,
				"away": away,
				"home_score": 0,
				"away_score": 0,
				"description": f"Home: {home}",
				"time_elapsed": -0.1
			})

			new_rows.append({
				"game_id": game_id,
				"home": home,
				"away": away,
				"home_score": 0,
				"away_score": 0,
				"description": f"Away: {away}",
				"time_elapsed": -0.09
			})

			new_rows.append({
				"game_id": game_id,
				"home": home,
				"away": away,
				"home_score": self.df.iloc[max_idx].home_score,
				"away_score": self.df.iloc[max_idx].away_score,
				"description": f"END",
				"time_elapsed": self.df.iloc[max_idx].time_elapsed + 0.1
			})


			self._min_len = min_len
			self._max_len = max_len

		new_stuff = pd.DataFrame(new_rows)
		self.df = pd.concat([self.df, new_stuff])
		self.df.sort_values(by=['game_id', 'time_elapsed'])
		self.df = self.df.reindex()

		self.plays = list(self.df['description'].unique())
		self.plays.insert(0, '<PAD>')
		self.play_ids_map = dict([(val, idx) for idx, val in enumerate(self.plays)])

		self._ranges = self._generate_ranges(min_len, max_len)

	def _generate_ranges(self, min_len, max_len):
		start = 0
		end = np.random.randint(min_len, max_len)

		ranges = [(start, end)]

		while end < (len(self.df) - max_len):
			start = end
			end = start + np.random.randint(min_len, max_len)

			while self.df.iloc[end].game_id != self.df.iloc[end].game_id:
				end -= 1

			if (end - start) < 1:
				continue

			ranges.append((start, end))

		return ranges

	def _team_ids_to_tensor(self, team_ids):
		tens = torch.zeros(len(self.teams))
		tens[team_ids[0]] = 1.
		tens[team_ids[1]] = 1.
		return tens

	def _teams_to_tensor(self, home, away):
		home_id = self.teams_id_map[home]
		away_id = self.teams_id_map[away]

		return self._team_ids_to_tensor([home_id, away_id])

	def __len__(self):
		return len(self._ranges)

	def _X_Y(self, subgame):
		#Y = torch.tensor([self.play_ids_map[subgame.iloc[-1].description]])
		X = torch.zeros(len(subgame)-1, len(self.plays))
		Y = torch.zeros(len(subgame)-1, len(self.plays))

		for X_idx, (_, play) in enumerate(subgame.iterrows()):
			if X_idx != (len(subgame) - 1):
				idx = self.play_ids_map[play.description]
				X[X_idx, idx] = 1.

			if X_idx != 0:
				idx = self.play_ids_map[play.description]
				Y[X_idx-1, idx] = 1.

		return X, Y

	def _subgame_to_tensor(self, subgame):
		X = torch.zeros(len(subgame), len(self.plays))
		for X_idx, (_, play) in enumerate(subgame.iterrows()):
			idx = self.play_ids_map[play.description]
			X[X_idx, idx] = 1.

		return X

	def __getitem__(self, idx):
		start_idx, end_idx = self._ranges[idx]

		subgame = self.df.iloc[start_idx:end_idx]
		X_plays, Y = self._X_Y(subgame)

		teams = [subgame.iloc[0]['home'], subgame.iloc[0]['away']]
		team_ids = [self.teams_id_map[t] for t in teams]
		teams_tens = self._team_ids_to_tensor(team_ids)

		return teams_tens, X_plays, Y
