import pandas as pd
import torch
import numpy as np
import math
from torch.utils.data import Dataset

class PlayIterator:
	def __init__(self, plays, times, sublen):
		self._cur_idx = 0
		self._sublen = sublen
		self._plays = plays
		self._times = times

	def __iter__(self):
		return self

	def __len__(self):
		return int(np.ceil(len(self._plays) / self._sublen))

	def __next__(self):
		if self._cur_idx + self._sublen + 1 < len(self._plays):
			end_idx = self._cur_idx + self._sublen + 1
		else:
			end_idx = len(self._plays)

		if self._cur_idx >= end_idx-1:
			raise StopIteration

		X_plays = self._plays[self._cur_idx : end_idx-1]
		Y_plays = self._plays[self._cur_idx+1 : end_idx]

		X_times = self._times[self._cur_idx : end_idx-1]
		Y_times = self._times[self._cur_idx+1 : end_idx]
		self._cur_idx = end_idx-1

		return X_plays, Y_plays, X_times, Y_times

class PlayByPlayDataset(Dataset):
	def __init__(self, df_path, subgame_len, limit_size=None):
		self.df = pd.read_csv('cleaned_data.csv')
		homes = self.df['home'].unique()
		aways = self.df['away'].unique()
		self.teams = sorted(list(set(homes).union(set(aways))))
		self._n_games = self.df['game_id'].nunique()

		self.teams_id_map = dict([(val, idx) for idx, val in enumerate(self.teams)])
		self.game_id_map = (
			dict(
				enumerate(
					list(self.df['game_id'].unique())
				)
			)
		)

		self._limit_size = limit_size

		new_rows = []
		self._subgame_len = subgame_len

		self.plays = sorted(list(self.df['description'].unique()))
		self.plays.insert(0, '<PAD>')
		self.play_ids_map = dict([(val, idx) for idx, val in enumerate(self.plays)])

	def _team_ids_to_tensor(self, team_ids):
		tens = torch.zeros(len(self.teams))
		tens[team_ids[0]] = 1.
		tens[team_ids[1]] = 1.
		return tens

	def _teams_to_tensor(self, home, away):
		home_id = self.teams_id_map[home]
		away_id = self.teams_id_map[away]

		return torch.tensor([home_id, away_id]).long()

	def __len__(self):
		if self._limit_size:
			return self._limit_size
		return self._n_games - 1

	def _X_Y(self, subgame):
		X_times = torch.tensor([r.time_elapsed for _, r in subgame[:-1].iterrows()])
		X = torch.tensor([self.play_ids_map[r.description] for _, r in subgame[:-1].iterrows()])
		Y = torch.tensor([self.play_ids_map[r.description] for _, r in subgame[1:].iterrows()])
		Y_times = torch.tensor([r.time_elapsed for _, r in subgame[1:].iterrows()])

		return X, Y, X_times, Y_times

	def __getitem__(self, idx):
		if idx >= len(self):
			raise StopIteration

		game_id = self.game_id_map[idx]

		game = self.df[self.df['game_id'] == game_id]
		X_plays, Y_plays, X_times, Y_times, = self._X_Y(game)

		teams = [game.iloc[0]['home'], game.iloc[0]['away']]
		team_ids = [self.teams_id_map[t] for t in teams]
		teams_tens = self._teams_to_tensor(teams[0], teams[1])

		times = torch.tensor([r.time_elapsed for _, r in game.iterrows()])
		plays = torch.tensor([self.play_ids_map[r.description] for _, r in game.iterrows()])

		return teams_tens, plays
