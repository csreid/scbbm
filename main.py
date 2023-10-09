import click
from tqdm import tqdm

@click.command
@click.option('--epochs', default=1, help='Number of epochs', type=int)
@click.option('--batch-size', default=1, help='Size of batch', type=int)
@click.option('--dataset-size', required=True, help='Number of samples to draw from play-by-play data', type=int)
@click.option('--data', default=None, help='Path to training data', required=True)
@click.option('--output-file', default=None, help='Path to save the output weights')
@click.option('--load-from', default=None, help='Checkpoint from which to resume training')
@click.option('--save-after-epoch/--no-save-after-epoch', default=True, help='Checkpoint after every epoch or only at the end?')
def main(epochs, batch_size, data, outputfile, load_from):
	import torch
	from models.game_sim import GameSimulator
	from trainers.sim_trainer import do_epoch
	from pbp_dataset import PlayByPlayDataset
	from torch.optim import Adam, SGD

	pbp = PlayByPlayDataset(load_from, min_len=2, max_len=20, size=100000)
	model = GameSimulator(len(pbp.teams), len(pbp.plays)).to('cuda:0')
	writer = SummaryWriter()
	opt = Adam(model.parameters())

	if load_from:
		try:
			model.load_state_dict(torch.load(load_from))
		except Exception as e:
			tqdm.write(f'Couldn\'t load model: {e}')

	epoch_prog = tqdm(range(epochs), position=0)
	all_steps = 0

	for epoch in epoch_prog:
		model, losses, steps = do_epoch(
			model,
			dataset,
			batch_size,
			opt,
			save_to=save_to,
			writer
		)

		all_steps += steps
