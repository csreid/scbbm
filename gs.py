import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import time
from pbp_dataset import PlayByPlayDataset
from score_dataset import ScoreDataset
from final_score_dataset import FinalScoreDataset
import torch
import pandas as pd
from torch.nn import GRU, CrossEntropyLoss, Module, KLDivLoss, LSTM, Linear, Embedding, functional as F, RNN, MSELoss, BCEWithLogitsLoss, L1Loss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import embed
import matplotlib.pyplot as plt
from itertools import chain

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def data_collate(samples):
	teams = [s[0] for s in samples]
	plays = [s[1] for s in samples]
	play_targets = [s[2] for s in samples]
	times = [s[3] for s in samples]
	time_targets = [s[4] for s in samples]

	teams_tens = torch.stack(teams)

	times_tens = pad_sequence(times)
	time_targets_tens = pad_sequence(time_targets)

	plays_tens = pad_sequence(plays)

	targets_tens = pad_sequence(play_targets)

	return teams_tens.long(), plays_tens.long(), targets_tens, times_tens.unsqueeze(2), time_targets_tens.unsqueeze(2)

def collate_scores(samples):
	teams = [s[0] for s in samples]
	scores_and_times = [s[1] for s in samples]

	scores_and_times = pad_sequence(scores_and_times)

	teams = torch.stack(teams)

	return teams, scores_and_times

class GameSimulator(Module):
	def __init__(self, n_teams, n_plays):
		super().__init__()
		self._n_plays = n_plays
		self._n_teams = n_teams

		team_feats = 10
		play_feats = 200

		self.team_input = Embedding(n_teams, team_feats, max_norm=1.)
		self.play_input = Embedding(n_plays, play_feats, max_norm=1., padding_idx=0)

		self.recurrent_score = LSTM(team_feats, 500, num_layers=1)
		self.score_out = Linear(500, 2)
		self.score_time_out = Linear(500, 1)

		self.final_score_out = Linear(team_feats, 2)

		self.recurrent1 = LSTM(2 * team_feats + play_feats + 1, 500, num_layers=2)
		self.recurrent2 = LSTM(500, 500, num_layers=1)
		self.play_fc = Linear(500, 500)
		self.play_out = Linear(500, n_plays)
		self.time_output = Linear(500, 1)

		self.team_filter = Linear(n_teams, n_plays)
		torch.nn.init.zeros_(self.team_filter.weight)

	def forward(self, teams, plays, times):
		batch_n = plays.shape[1]
		seq_n = plays.shape[0]

		play_emb = self.play_input(plays)

		with torch.no_grad():
			flat_teams = torch.flatten(self.team_input(teams), start_dim=1, end_dim=2)

		teams_emb = (flat_teams).unsqueeze(0).expand(seq_n, -1, -1)

		out = torch.cat((teams_emb, play_emb, times), dim=2)
		out, _ = self.recurrent1(out)

		out1 = self.play_fc(out)
		out1 = self.play_out(out)

		out2, _ = self.recurrent2(out)
		out2  = self.time_output(out2)

		dev = next(self.parameters()).device
		team_filter_input = torch.zeros(len(teams), self._n_teams)
		for (h_idx, a_idx), row in zip(teams, team_filter_input):
			row[h_idx] = 1.
			row[a_idx] = 1.

		team_filter_input = team_filter_input.expand(1, len(teams), self._n_teams).to(dev)
		filter_result = self.team_filter(team_filter_input)

		out1 = out1 * filter_result

		return out1, out2

	def scores(self, teams, length=200):
		with torch.no_grad():
			teams = self.team_input(teams)
		teams = torch.diff(teams, dim=1)
		teams = teams.squeeze(1)

		batch_size, team_feats = teams.shape

		teams = teams.expand(length, batch_size, team_feats)

		out, _ = self.recurrent_score(teams)
		out = F.sigmoid(out)

		score_out = F.sigmoid(self.score_out(out)) * 200
		time_out = F.sigmoid(self.score_time_out(out)) * 3600.

		return score_out, time_out

	def final_score(self, teams):
		out = self.team_input(teams)
		out = torch.diff(out, dim=1)
		out = out.squeeze(1)
		out = self.final_score_out(out)

		return out

	def sim_scores(self, home, away, ds, length):
		teams_tens = ds._teams_to_tensor(home, away)

		scores, times = self.scores(teams_tens.unsqueeze(0), length)

		return scores, times

	def simulate(self, home, away, dataset, maxlen=10):
		plays = ['Jump Ball won by Purdue']
		times = [0]
		teams_tens = dataset._teams_to_tensor(home, away)

		if maxlen:
			condition = lambda idx: idx == maxlen
		else:
			condition = lambda idx: (plays[-1] == 'End of Game') or (idx == 300)

		idx = 0
		while not condition(idx):
			subgame = pd.DataFrame({"description": plays, "time_elapsed": times})
			plays_tens, times_tens = dataset._subgame_to_tensor(subgame)

			with torch.no_grad():
				play_out, time_out = self.forward(teams_tens.unsqueeze(0), plays_tens.unsqueeze(1), times_tens.reshape((-1, 1, 1)))
				play_out = play_out[-1, 0, :]
				next_play_dist = F.softmax(play_out[1:], dim=0).numpy()
				next_time = time_out[-1]

			next_play_idx = np.random.choice(self._n_plays-1, p=next_play_dist) + 1
			next_play = dataset.plays[next_play_idx]

			times.append(next_time)
			plays.append(next_play)
			idx += 1

		return plays, times


def main_sim(gs):
	pbp = PlayByPlayDataset('cleaned_data.csv', min_len=2, max_len=20, size=100000)

	batch_size=32
	loader = DataLoader(pbp, batch_size=batch_size, collate_fn=data_collate, shuffle=True)

	play_loss_fn = CrossEntropyLoss(reduction='mean', ignore_index=0)
	time_loss_fn = L1Loss()
	opt = Adam(gs.parameters())

	filter_opt = SGD(gs.team_filter.parameters(), lr=0.2)

	epoch_prog = tqdm(range(500), position=0)
	tqdm.write(f'Length of dataset: {len(pbp)}')
	counter = 0
	for epoch in epoch_prog:
		prog = tqdm(enumerate(loader), total=int(len(pbp) / batch_size), position=1, leave=False)

		for idx, (team_X, play_X, play_Y, time_X, time_Y) in prog:
			team_X = team_X.to('cuda:0')
			play_X = play_X.to('cuda:0')

			Y = play_Y.to('cuda:0').transpose(0,1)

			time_Y = time_Y.to('cuda:0')
			time_X = time_X.to('cuda:0')

			Y_pred_play, Y_pred_time = gs(team_X, play_X, time_X)

			Y_pred_play = Y_pred_play.to('cuda:0').transpose(0,2).transpose(0,1)
			Y_pred_time = Y_pred_time.to('cuda:0')

			play_loss = play_loss_fn(Y_pred_play, Y)
			time_loss = time_loss_fn(Y_pred_time, time_Y)
			loss = play_loss + time_loss

			writer.add_scalar('Loss/play', play_loss, counter)
			writer.add_scalar('Loss/time', time_loss, counter)
			writer.add_scalar('Loss/total', loss, counter)

			filter_opt.zero_grad()
			opt.zero_grad()
			loss.backward()
			opt.step()
			filter_opt.step()

			counter += 1

		#torch.save(gs.state_dict(), 'model.ptch')
		# Do a test run
		#gs = gs.to('cpu')
		#plays, times = gs.simulate('Purdue', 'Rutgers', pbp, maxlen=None)
		#sample = "  \n".join([f'{p} @ {t} seconds' for p, t in zip(plays, times)])
		#writer.add_text(f'Sample', sample, epoch * (len(pbp) / batch_size))
		#gs = gs.to('cuda:0')

	return model

def main_score(gs):
	sd = ScoreDataset('cleaned_data.csv')

	batch_size=128

	loader = DataLoader(sd, batch_size=batch_size, collate_fn=collate_scores, shuffle=True)

	loss_fn = MSELoss()
	opt = Adam(gs.parameters())

	epoch_prog = tqdm(range(500), position=0)
	ctr = 0

	for epoch in epoch_prog:
		prog = tqdm(loader, total=int(len(sd) / batch_size), position=1, leave=False)
		for X, Y in prog:
			X = X.to('cuda:0')
			Y = Y.to('cuda:0')
			Y_pred_scores, Y_pred_times = gs.scores(X, length=Y.shape[0])
			Y_pred_scores = Y_pred_scores.to('cuda:0')
			Y_pred_times = Y_pred_times.to('cuda:0')

			loss_scores = loss_fn(Y_pred_scores, Y[:, :, :2])# + torch.sum((Y_pred_scores - torch.abs(Y_pred_scores)) ** 2)
			loss_time = loss_fn(Y_pred_times, Y[:, :, 2:])

			loss = loss_scores + loss_time

			opt.zero_grad()
			loss.backward()
			opt.step()

			writer.add_scalar('Loss/score', loss_scores, ctr)
			writer.add_scalar('Loss/time', loss_time, ctr)
			ctr += 1
			purdue_id = sd.teams_id_map['Purdue']

			if (ctr % 100) == 0:
				gs.to('cpu')
				with torch.no_grad():
					scores, times = gs.sim_scores('Purdue', 'UConn', sd, 100)
					home_score = scores[:, 0, 0].detach().numpy()
					away_score = scores[:, 0, 1].detach().numpy()
					times = times[:, 0].detach().numpy()

					fig = plt.figure()
					plt.plot(times, home_score, label='Purdue')
					plt.plot(times, away_score, label='UConn')
					plt.legend()

					writer.add_figure('Purdue v. UConn', fig, ctr)

					fig = plt.figure()
					purdue_vals = gs.team_input(torch.tensor([sd.teams_id_map['Purdue']]))
					plt.bar(np.arange(10), purdue_vals.detach().numpy().squeeze())
					plt.ylim(-1,  1)
					writer.add_figure('Purdue Traits', fig, ctr)

					embs = gs.team_input(torch.tensor([sd.teams_id_map[t] for t in sd.teams]))
					metadata = sorted(sd.teams)
					writer.add_embedding(embs, metadata=metadata, global_step=ctr, tag='Team Embeddings')

					gs.to('cuda:0')
		torch.save(gs.state_dict(), 'model.ptch')

	return gs

def main_finalscore(gs):
	batch_size=128

	sd = FinalScoreDataset('cleaned_data.csv')
	loader = DataLoader(sd, batch_size=batch_size, shuffle=True)

	loss_fn = MSELoss()
	opt = Adam(gs.parameters())

	epoch_prog = tqdm(range(1000), position=0)
	ctr = 0

	for epoch in epoch_prog:
		prog = tqdm(loader, total=int(len(sd) / batch_size), position=1, leave=False)
		for X, Y in prog:
			X = X.to('cuda:0')
			Y = Y.to('cuda:0')

			Y_pred = gs.final_score(X)
			loss = loss_fn(Y_pred, Y)

			opt.zero_grad()
			loss.backward()
			opt.step()

			writer.add_scalar('Loss/final_score', loss, ctr)
			ctr += 1

		torch.save(gs.state_dict(), 'model.ptch')

	return gs

if __name__ == '__main__':
	pbp = PlayByPlayDataset('cleaned_data.csv', min_len=2, max_len=5, size=1000000)
	model = GameSimulator(len(pbp.teams), len(pbp.plays)).to('cuda:0')
	try:
		model.load_state_dict(torch.load('model.ptch'))
	except Exception as e:
		tqdm.write(f'Couldn\'t load model: {e}')

	model = main_finalscore(model)
	model = main_sim(model)
	torch.save(model.state_dict(), 'model.ptch')
