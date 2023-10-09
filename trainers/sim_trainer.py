import torch
import tqdm.

def _data_collate(samples):
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


def do_epoch(model, dataset, batch_size, optimizer, save_to=None, writer=None):
	loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_data_collate, shuffle=True)

	prog = tqdm(enumerate(loader), total=int(len(dataset) / batch_size), position=1, leave=False)

	steps = 0
	for idx, (team_X, play_X, play_Y, time_X, time_Y) in prog:
		team_X = team_X.to('cuda:0')
		play_X = play_X.to('cuda:0')

		Y = play_Y.to('cuda:0').transpose(0,1)

		time_Y = time_Y.to('cuda:0')
		time_X = time_X.to('cuda:0')

		Y_pred_play, Y_pred_time = model(team_X, play_X, time_X)

		Y_pred_play = Y_pred_play.to('cuda:0').transpose(0,2).transpose(0,1)
		Y_pred_time = Y_pred_time.to('cuda:0')

		play_loss = play_loss_fn(Y_pred_play, Y)
		time_loss = time_loss_fn(Y_pred_time, time_Y)
		loss = play_loss + time_loss

		filter_opt.zero_grad()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		filter_opt.step()

		steps += 1

	return model, [play_loss, time_loss, loss], steps
