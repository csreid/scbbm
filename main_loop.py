import torch
from fg_dataset import PlayByPlayDataset
from models.game_sim import GameSimulator
from torch.optim import Adam, SGD
from torch.nn import L1Loss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from collections import deque
from itertools import islice
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def _collate(batch):
	teams = [r[0] for r in batch]
	plays = [r[1] for r in batch]

	teams_tensor = torch.stack(teams, dim=0)

	plays_tensor = pad_sequence(plays)
	return teams_tensor.expand(plays_tensor.shape[0], -1, -1), plays_tensor


def fit_on_game(model, teams, plays, subgame_len=20, device="cuda:0"):
	state = None
	total_play_loss = 0
	total_time_loss = 0
	steps = 0
	teams = teams[0:subgame_len].to("cpu", non_blocking=True)
	substeps = len(plays) / subgame_len
	for play_idx in range(0, len(plays) - 1, subgame_len):
		x_end_idx = min(play_idx + subgame_len, len(plays) - 1)
		X_plays = plays[play_idx:x_end_idx].to("cpu", non_blocking=True)
		Y_plays = plays[play_idx + 1 : play_idx + subgame_len + 1].to(
			"cuda:0", non_blocking=True
		)

		if len(teams) > len(X_plays):
			teams = teams[: len(X_plays)]

		Y_pred_play, state = model(teams, X_plays, None, state=state)
		state = tuple([s.detach() for s in state])

		Y_pred_play = Y_pred_play.to(device, non_blocking=True)
		fullpred = torch.flatten(Y_pred_play, start_dim=0, end_dim=1)
		target = torch.flatten(Y_plays)

		play_loss = play_loss_fn(fullpred, target)
		loss = play_loss
		steps += 1
		total_play_loss += play_loss

		loss = loss / substeps

		loss.backward()
	opt.step()

	return total_play_loss / substeps


if __name__ == "__main__":
	ds = PlayByPlayDataset(df_path="foo", subgame_len=20)
	model = GameSimulator(len(ds.teams), len(ds.plays))
	play_loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=0)
	time_loss_fn = L1Loss()
	opt = SGD(model.parameters(), lr=1e-3)
	scheduler = ReduceLROnPlateau(opt, patience=0, min_lr=[1e-6])
	writer = SummaryWriter()
	batch_size = 32
	loader = DataLoader(
		ds, batch_size=batch_size, collate_fn=_collate, shuffle=True
	)

	model = model.to("cuda:0")

	prev_epoch_loss = np.inf
	ds_len = len(ds)
	losses = []

	for epoch in tqdm(range(100)):
		for idx, (teams, game) in tqdm(
			enumerate(loader), total=int(len(ds) / batch_size), leave=False
		):
			play_loss = fit_on_game(model, teams, game, subgame_len=50)
			total_loss = play_loss
			curstep = epoch * int((ds_len / batch_size)) + idx

			losses.append(total_loss.item())

			writer.add_scalar("loss/play", play_loss, curstep)

		writer.add_scalar("loss/avg", np.mean(losses), curstep)
		scheduler.step(np.mean(np.mean(losses)))
		losses = []
		torch.save(model.state_dict(), "model.ptch")
