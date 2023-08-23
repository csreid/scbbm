import pandas as pd
import torch
import numpy as np
import math
from torch.utils.data import Dataset

class PlayByPlayDataset(Dataset):
	def __init__(self, df_path, context_len=10):
		self.df = pd.read_csv('cleaned_data.csv')
		homes = self.df['home'].unique()
		aways = self.df['away'].unique()
		self.teams = list(set(homes).union(set(aways)))

		self.teams_id_map = dict([(val, idx) for idx, val in enumerate(self.teams)])
		self.game_id_map = (
			dict(
				enumerate(
					list(self.df['game_id'].unique())
				)
			)
		)

		self.plays = list(self.df['description'].unique())
		self.plays.insert(0, '<PAD>')
		self.play_ids_map = dict([(val, idx) for idx, val in enumerate(self.plays)])

		self._max_len = context_len#self.df.game_id.value_counts().max()
		self._n_games = self.df['game_id'].nunique()

	def _team_ids_to_tensor(self, team_ids):
		tens = torch.zeros(len(self.teams))
		tens[team_ids[0]] = 1.
		tens[team_ids[1]] = 1.
		return tens#.expand(self._max_len, len(self.teams))

	def _teams_to_tensor(self, home, away):
		home_id = self.teams_id_map[home]
		away_id = self.teams_id_map[away]

		return self._team_ids_to_tensor([home_id, away_id])

	def _game_idx_and_len(self, idx:int):
		game_idx = idx % self._n_games
		sublen = math.floor(idx / self._n_games)

		return game_idx, max(sublen, 4)

	def _Y_tensor(self, game):
		play_idx = self.play_ids_map[game.iloc[-1].description]
		play_tens = torch.tensor([play_idx])

		return play_tens

	def _get_start_and_end(self, idx):
		start_idx = idx

		start_game_id = self.df.iloc[idx].game_id
		end_idx = idx + self._max_len + 1

		while self.df.iloc[end_idx].game_id != start_game_id:
			end_idx -= 1

		return start_idx, end_idx

	def _X_tensor(self, game):
		plays = []

		play_tens = torch.zeros(self._max_len, len(self.plays), dtype=torch.long)
		for idx, play in enumerate(plays):
			if idx >= self._max_len:
				break
			play_idx = self.play_ids_map[play]
			play_tens[idx, play_idx] = 1.

		return play_tens

	def _plays_tensors(self, game, game_len):
		X = self._X_tensor(game, game_len)
		Y = self._Y_tensor(game)

		return X, Y

	def __len__(self):
		return self.df.game_id.nunique() * self._max_len

	def __getitem__(self, idx):
		game_idx, game_len = self._game_idx_and_len(idx)
		game_id = self.game_id_map[game_idx]

		game = self.df[self.df.game_id == game_id][:game_len+1]

		plays_tens, target_play_tens = self._plays_tensors(game, game_len)

		teams = [game.iloc[0]['home'], game.iloc[0]['away']]
		team_ids = [self.teams_id_map[t] for t in teams]
		teams_tens = self._team_ids_to_tensor(team_ids)

		return teams_tens, plays_tens, target_play_tens
