import click
from tqdm import tqdm


@click.command()
@click.option("--epochs", default=1, help="Number of epochs", type=int)
@click.option("--batch-size", default=16, help="Games per batch", type=int)
@click.option(
	"--data",
	default=None,
	help="Path to a pre-built CSV or parquet",
	type=click.Path(exists=True),
)
@click.option(
	"--data-dir",
	default=None,
	help="Path to ncaahoopR_data directory",
	type=click.Path(exists=True, file_okay=False),
)
@click.option(
	"--seasons",
	default=None,
	help='Comma-separated seasons to include, e.g. "2022-23,2023-24". Default: all.',
	type=str,
)
@click.option(
	"--cache",
	default="pbp_cache.parquet",
	show_default=True,
	help="Parsed parquet cache. Read if present, written on first run.",
	type=str,
)
@click.option(
	"--val-fraction",
	default=0.05,
	show_default=True,
	help="Fraction of games held out for validation.",
	type=float,
)
@click.option(
	"--sim-steps",
	default=30,
	show_default=True,
	help="Plays to generate in the TensorBoard simulation log.",
	type=int,
)
@click.option(
	"--output-file", default=None, help="Path to save model weights", type=str
)
@click.option(
	"--load-from", default=None, help="Checkpoint to resume from", type=str
)
@click.option("--save-after-epoch/--no-save-after-epoch", default=True)
@click.option(
	"--eval-every",
	default=5,
	show_default=True,
	help="Compute domain-specific game quality metrics every N epochs.",
	type=int,
)
@click.option(
	"--eval-games",
	default=20,
	show_default=True,
	help="Number of games to simulate for domain metric evaluation.",
	type=int,
)
@click.option(
	"--eval-steps",
	default=200,
	show_default=True,
	help="Play steps per simulated game during evaluation.",
	type=int,
)
def main(
	epochs,
	batch_size,
	data,
	data_dir,
	seasons,
	cache,
	val_fraction,
	sim_steps,
	output_file,
	load_from,
	save_after_epoch,
	eval_every,
	eval_games,
	eval_steps,
):
	import os
	import torch
	from models.game_sim import GameSimulator
	from trainers.sim_trainer import do_epoch, do_validation
	from pbp_dataset import PlayByPlayDataset, build_dataframe, train_val_split
	from game_metrics import (
		build_reference_stats,
		simulate_n_games,
		compute_metrics,
		log_metrics,
	)
	from torch.optim import Adam
	from torch.utils.tensorboard import SummaryWriter

	if data is None and data_dir is None:
		raise click.UsageError("Provide either --data or --data-dir.")

	device = "cuda:0" if torch.cuda.is_available() else "cpu"

	if data_dir:
		raw_cache = cache.replace(".parquet", "_raw.parquet")
		if not os.path.exists(raw_cache):
			season_list = seasons.split(",") if seasons else None
			df = build_dataframe(data_dir, seasons=season_list)
			df.to_parquet(raw_cache, index=False)
			tqdm.write(f"Raw data written to {raw_cache}")
		data = raw_cache

	dataset = PlayByPlayDataset(data, cache_path=cache)
	train_set, val_set = train_val_split(dataset, val_fraction=val_fraction)
	tqdm.write(
		f"{len(train_set)} train games | {len(val_set)} val games | "
		f"{len(dataset.teams)} teams | {len(dataset.players)} players"
	)

	model = GameSimulator(
		n_teams=len(dataset.teams),
		n_play_types=len(dataset.play_types),
		n_players=len(dataset.players),
	).to(device)

	if load_from:
		try:
			model.load_state_dict(torch.load(load_from, map_location=device))
			tqdm.write(f"Loaded weights from {load_from}")
		except Exception as e:
			tqdm.write(f"Could not load weights: {e}")

	optimizer = Adam(model.parameters())
	writer = SummaryWriter()
	all_steps = 0

	tqdm.write("Building reference statistics from training set...")
	ref_stats = build_reference_stats(train_set)

	# Use a fixed val game index for the simulation log so it's comparable across epochs
	sim_game_idx = 0

	for epoch in tqdm(range(epochs), position=0):
		model, train_losses, steps = do_epoch(
			model,
			train_set,
			batch_size,
			optimizer,
			device=device,
			writer=writer,
			start_step=all_steps,
		)
		all_steps += steps

		val_losses = do_validation(
			model,
			val_set,
			batch_size,
			device=device,
			writer=writer,
			epoch=epoch,
		)

		tqdm.write(
			f"Epoch {epoch + 1}: "
			f"train={train_losses['total']:.4f}  val={val_losses['total']:.4f}  "
			f"acc={train_losses['play_type_accuracy']:.3f}/{val_losses['play_type_accuracy']:.3f}  "
			f"pplx={torch.tensor(train_losses['play_type']).exp().item():.2f}/"
			f"{torch.tensor(val_losses['play_type']).exp().item():.2f}"
		)

		# --- Simulated game log ---
		home, away, sim = model.simulate_game(
			val_set, sim_game_idx, n_steps=sim_steps, device=device
		)
		rows = [
			"| # | Pred type | Pred player | True type | True player |",
			"|---|-----------|-------------|-----------|-------------|",
		]
		for i, (pt, pp, tt, tp) in enumerate(sim, 1):
			match = "✓" if pt == tt else " "
			rows.append(f"| {i} | {pt} | {pp} | {tt} {match} | {tp} |")
		writer.add_text(f"sim/{home}_vs_{away}", "\n".join(rows), epoch)

		# Team embedding projector
		team_indices = torch.arange(len(dataset.teams), device=device)
		team_embs = model.team_input(team_indices).detach().cpu()
		writer.add_embedding(
			team_embs, metadata=dataset.teams, global_step=epoch, tag="teams"
		)

		# --- Domain-specific game quality metrics ---
		if eval_every > 0 and (epoch + 1) % eval_every == 0:
			tqdm.write(
				f"  Running game quality metrics ({eval_games} games × {eval_steps} steps)..."
			)
			sim_games = simulate_n_games(
				model, val_set, eval_games, n_steps=eval_steps, device=device
			)
			metrics = compute_metrics(sim_games, ref_stats)
			log_metrics(metrics, writer, epoch)
			tqdm.write(
				f"  membership={metrics['metrics/team_membership_acc']:.3f}  "
				f"reb_after_miss={metrics['metrics/rebound_after_miss_rate']:.3f}"
				f"(ref={metrics['metrics_ref/rebound_after_miss']:.3f})  "
				f"sub_pair={metrics['metrics/sub_pairing_rate']:.3f}"
				f"(ref={metrics['metrics_ref/sub_pairing']:.3f})  "
				f"ft_adj={metrics['metrics/ft_adjacency_rate']:.3f}"
				f"(ref={metrics['metrics_ref/ft_adjacency']:.3f})  "
				f"box_outlier={metrics['metrics/box_score_outlier_rate']:.3f}"
			)

		if save_after_epoch and output_file:
			torch.save(model.state_dict(), output_file)
			tqdm.write(f"  saved to {output_file}")


if __name__ == "__main__":
	main()
