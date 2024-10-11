import pandas as pd
import torch
import numpy as np
import math
from torch.utils.data import Dataset

class PlayByPlayDataset(Dataset):
	def __init__(self, df_path, min_len, max_len, size):
		self.df = pd.read_csv('cleaned_data.csv')
		#self.df = self.df[(self.df.home == 'Purdue') | (self.df.away == 'Purdue')]
		homes = self.df['home'].unique()
		aways = self.df['away'].unique()
		self.teams = list(set(homes).union(set(aways)))
		self._n_games = self.df['game_id'].nunique()
		self._size = size
		self._min_len = min_len
		self._max_len = max_len

		self.teams_id_map = dict([(val, idx) for idx, val in enumerate(self.teams)])
		self.game_id_map = (
			dict(
				enumerate(
					list(self.df['game_id'].unique())
				)
			)
		)

		new_rows = []

		self.plays = sorted(list(self.df['description'].unique()))
		self.plays.insert(0, '<PAD>')
		self.play_ids_map = dict([(val, idx) for idx, val in enumerate(self.plays)])

		self._ranges = dict()

	def _generate_range(self):
		length = np.random.randint(self._min_len, self._max_len)
		start = np.random.randint(len(self.df) - length)
		end = start + length

		try:
			while self.df.iloc[start].game_id != self.df.iloc[end].game_id:
				start -= 1
				end -= 1

		except IndexError:
			print(f'Index {end} was (probably) OOB ({len(self.df)})')
		return (start, end)


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
		return self._size

	def _X_Y(self, subgame):
		X_times = torch.tensor([r.time_elapsed for _, r in subgame[:-1].iterrows()])
		X = torch.tensor([self.play_ids_map[r.description] for _, r in subgame[:-1].iterrows()])
		Y = torch.tensor([self.play_ids_map[r.description] for _, r in subgame[1:].iterrows()])
		Y_times = torch.tensor([r.time_elapsed for _, r in subgame[1:].iterrows()])

		return X, Y, X_times, Y_times

	def _subgame_to_tensor(self, subgame):
		X = torch.tensor([self.play_ids_map[r.description] for _, r in subgame.iterrows()])
		X_times = torch.tensor([r.time_elapsed for _, r in subgame.iterrows()])
		return X, X_times

	def __getitem__(self, idx):
		if idx in self._ranges:
			start_idx, end_idx = self._ranges[idx]
		else:
			start_idx, end_idx = self._generate_range()
			self._ranges[idx] = (start_idx, end_idx)

		subgame = self.df.iloc[start_idx:end_idx]
		X_plays, Y_plays, X_times, Y_times, = self._X_Y(subgame)

		teams = [subgame.iloc[0]['home'], subgame.iloc[0]['away']]
		team_ids = [self.teams_id_map[t] for t in teams]
		teams_tens = self._teams_to_tensor(teams[0], teams[1])

		return teams_tens, X_plays, Y_plays, X_times, Y_times
