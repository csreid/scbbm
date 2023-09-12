import pandas as pd
import torch
import numpy as np
import math
from torch.utils.data import Dataset
import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

class ScoreDataset(Dataset):
	def __init__(self, df_path):
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
		Y_home = torch.tensor(list(subgame.home_score))
		Y_away = torch.tensor(list(subgame.away_score))
		Y_time = torch.tensor(list(subgame.time_elapsed))

		Y = torch.stack([Y_home, Y_away, Y_time], dim=0)
		Y = Y.transpose(0, 1)

		return X, Y

	def __getitem__(self, idx):
		game_id = self.game_id_map[idx]

		X, Y = self._X_Y(self.df[self.df.game_id == game_id])

		return X, Y

if __name__ == '__main__':
	from torch.utils.data import DataLoader

	sd = ScoreDataset('cleaned_data.csv')
	loader = DataLoader(sd)

	print(next(iter(loader)))
