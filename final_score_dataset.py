import pandas as pd
import torch
import numpy as np
import math
from torch.utils.data import Dataset
import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

class FinalScoreDataset(Dataset):
	def __init__(self, df_path):
		self.df = pd.read_csv('cleaned_data.csv')
		homes = self.df['home'].unique()
		aways = self.df['away'].unique()
		self.teams = list(set(homes).union(set(aways)))
		self._n_games = self.df['game_id'].nunique()
		self.plays = sorted(list(self.df['description'].unique()))

		self.teams_id_map = dict([(val, idx) for idx, val in enumerate(sorted(self.teams))])
		self.game_id_map = (
			dict(
				enumerate(
					list(self.df['game_id'].unique())
				)
			)
		)

		new_rows = []

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
		return len(self.game_id_map)

	def _X_Y(self, subgame):
		# X = (home_id, away_id)
		# Y = <sequence of scores for home & away, + time>

		home = subgame.iloc[0].home
		away = subgame.iloc[0].away

		X = self._teams_to_tensor(home, away)
		Y = torch.tensor([subgame.iloc[-1].home_score, subgame.iloc[-1].away_score])

		return X, Y.float()

	def __getitem__(self, idx):
		game_id = self.game_id_map[idx]
		subgame = self.df[self.df.game_id == game_id]

		return self._X_Y(subgame)
