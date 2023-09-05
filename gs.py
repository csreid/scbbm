import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
from pbp_dataset import PlayByPlayDataset
import torch
import pandas as pd
from torch.nn import GRU, CrossEntropyLoss, Module, KLDivLoss, LSTM, Linear, functional as F, RNN, MSELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import embed

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

	targets_tens = pad_sequence(targets)

	#targets_tens = torch.stack(targets_tens)
	targets_tens = pack_padded_sequence(targets_tens, [len(p) for p in targets], enforce_sorted=False)

	return teams_tens.unsqueeze(0), plays_tens, targets_tens#targets_tens.squeeze()


class GameSimulator(Module):
	def __init__(self, n_teams, n_plays):
		super().__init__()
		self._n_plays = n_plays

		self.team_input = Linear(n_teams, 1000)
		self.play_input = Linear(n_plays, 1000)
		self.recurrent = GRU(1000, 1000)
		self.output = Linear(1000, n_plays)

	def forward(self, teams, plays):
		team_out = F.tanh(self.team_input(teams))
		team_out, h = self.recurrent(team_out)

		play_out = F.tanh(self.play_input(plays))
		out, _ = self.recurrent(play_out, h)
		#out = out[-1, :, :]
		out = self.output(out)
		#out = out.transpose(0, 1)

		out_shape = out.shape

		return out

	def simulate(self, home, away, dataset, maxlen=10):
		plays = [f'Home: {home}', f'Away: {away}']
		teams_tens = dataset._teams_to_tensor(home, away)

		for _ in range(maxlen):
			subgame = pd.DataFrame({"description": plays})
			plays_tens = dataset._subgame_to_tensor(subgame)
			with torch.no_grad():
				next_play_dist = F.softmax(self.forward(teams_tens.unsqueeze(0).unsqueeze(0), plays_tens.unsqueeze(1))[-1, :].squeeze()).numpy()
	
			next_play_idx = np.random.choice(self._n_plays, p=next_play_dist)
			next_play = dataset.plays[next_play_idx]

			plays.append(next_play)

		return plays


if __name__ == '__main__':
	try:
		pbp = PlayByPlayDataset('cleaned_data.csv', min_len=3, max_len=5)
		gs = GameSimulator(len(pbp.teams), len(pbp.plays)).to('cuda:0')
		try:
			gs.load_state_dict(torch.load('model.ptch'))
			print(f'Successfully loaded model')
		except:
			print('Could not resume model training')

		finally:
			gs.to('cuda:0')
			pass
		batch_size=32
		loader = DataLoader(pbp, batch_size=batch_size, collate_fn=data_collate, shuffle=True)

		play_loss_fn = CrossEntropyLoss(reduction='mean')
		opt = Adam(gs.parameters())

		losses = []
		epoch_prog = tqdm(range(10), position=0)
		print(f'Length of dataset: {len(pbp)}')
		for epoch in epoch_prog:
			if len(losses) != 0:
				epoch_prog.set_postfix({'Previous epoch loss': np.mean(losses)})
			losses = []
			prog = tqdm(enumerate(loader), total=len(pbp) / batch_size, position=1, leave=False)
			for idx, (team_X, play_X, Y) in prog:
				team_X = team_X.to('cuda:0')
				play_X = play_X.to('cuda:0')
				play_X = pad_packed_sequence(play_X)[0].float()
	
				Y = pad_packed_sequence(Y)[0].float()
				Y = Y.transpose(0, 1).transpose(1,2)
				Y = Y.to('cuda:0')

				Y_pred = gs(team_X, play_X).to('cuda:0').transpose(0,2).transpose(0,1)
				loss = play_loss_fn(Y_pred, Y)
				losses.append(loss.item())
				writer.add_scalar('Loss/train', loss, epoch * (len(pbp)/batch_size) + idx)

				opt.zero_grad()
				loss.backward()
				opt.step()
				#epoch_prog.set_description(f'Step {idx}/{len(loader)}')
				if (idx % 1000) == 0:
					gs = gs.to('cpu')
					sample = "\n".join(gs.simulate("Purdue", "UConn", pbp))
					writer.add_text('sample', sample, idx)
					gs = gs.to('cuda:0')

			torch.save(gs.state_dict(), 'model.ptch')
	except:
		raise
		#embed()
