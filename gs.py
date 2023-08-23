import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')

from pbp_dataset import PlayByPlayDataset
import torch
from torch.nn import GRU, CrossEntropyLoss, Module, KLDivLoss, LSTM, Linear, functional as F, RNN, MSELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def data_collate(samples):
	teams = [s[0] for s in samples]
	plays = [s[1] for s in samples]
	targets = [s[2] for s in samples]

	teams_tens = torch.stack(teams)

	plays_tens = pad_sequence(plays)
	plays_tens = pack_padded_sequence(plays_tens, [len(p) for p in plays], enforce_sorted=False)

	targets_tens = torch.stack(targets)

	return teams_tens, plays_tens, targets_tens.squeeze()


class GameSimulator(Module):
	def __init__(self, n_teams, n_plays, max_len):
		super().__init__()
		self._n_plays = n_plays
		self._max_len = max_len

		## With the "predict plays and the time they occurred" model, use a 
		## 2-layer GRU/LSTM/whatever. The first layer should be used to predict the play,
		## and the second layer should be used to predict the time it happened

		self.team_input = Linear(n_teams, 1000)
		self.play_input = Linear(n_plays, 1000)
		self.recurrent = LSTM(1000, 1000, batch_first=True)
		self.output = Linear(1000, n_plays)

	def forward(self, teams, plays):
		team_out = self.team_input(teams)
		play_out = self.play_input(plays)

		out, (h, c) = self.recurrent(team_out)
		out, (h, c) = self.recurrent(out, (h, c))
		out = self.output(out)

		return out

pbp = PlayByPlayDataset('cleaned_data.csv', context_len=4)
gs = GameSimulator(len(pbp.teams), len(pbp.plays), pbp._max_len).to('cuda:0')
batch_size=32
loader = DataLoader(pbp, batch_size=batch_size, collate_fn=data_collate)

play_loss_fn = CrossEntropyLoss(reduction='mean')
opt = Adam(gs.parameters())

gs.to('cuda:0')
losses = []
epoch_prog = tqdm(range(10), desc=" epoch", position=0)
#prog = tqdm(enumerate(loader), total=len(pbp) / batch_size, position=1, leave=False)
for epoch in epoch_prog:#range(10):#epoch_prog:
	for idx, (team_X, play_X, Y) in enumerate(loader):#prog:
		team_X = team_X.to('cuda:0')
		play_X = play_X.to('cuda:0')
		play_X = pad_packed_sequence(play_X)[0].float()


		Y = Y.to('cuda:0')

		Y_pred = gs(team_X, play_X).to('cuda:0')

		loss = play_loss_fn(Y_pred, Y)
		losses.append(loss.item())
		writer.add_scalar('Loss/train', loss, epoch * (len(pbp)/batch_size) + idx)

		opt.zero_grad()
		loss.backward()
		opt.step()
	epoch_prog.set_description(f'Loss: {np.mean(losses)}')
	losses = []
