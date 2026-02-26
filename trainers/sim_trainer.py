import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from pbp_dataset import collate_fn


def do_epoch(
	model,
	dataset,
	batch_size,
	optimizer,
	device="cuda:0",
	save_to=None,
	writer=None,
	start_step=0,
):
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		collate_fn=collate_fn,
		shuffle=True,
		num_workers=4,
		pin_memory=True,
	)

	play_type_loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=0)
	player_loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=0)

	total_play_type_loss = 0.0
	total_player_loss = 0.0
	total_correct = 0
	total_count = 0

	prog = tqdm(
		enumerate(loader),
		total=len(dataset) // batch_size,
		position=1,
		leave=False,
	)
	model.train()

	steps = 0
	for _, (teams, x_types, x_players, x_state, y_types, y_players) in prog:
		teams = teams.to(device)
		x_types = x_types.to(device)
		x_players = x_players.to(device)
		x_state = x_state.to(device)
		y_types = y_types.to(device)
		y_players = y_players.to(device)

		# Pass y_types as target_play_types so the player head is conditioned on
		# the correct next play type during training (teacher forcing).
		play_type_logits, player_logits, _ = model(
			teams, x_types, x_players, x_state, target_play_types=y_types
		)

		seq, batch = y_types.shape
		flat_y_types = y_types.view(seq * batch)
		flat_y_players = y_players.view(seq * batch)

		play_type_loss = play_type_loss_fn(
			play_type_logits.view(seq * batch, -1), flat_y_types
		)
		player_loss = player_loss_fn(
			player_logits.view(seq * batch, -1), flat_y_players
		)
		loss = play_type_loss + player_loss

		optimizer.zero_grad()
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			model.parameters(), max_norm=5.0
		)
		optimizer.step()

		mask = flat_y_types != 0
		if mask.any():
			preds = play_type_logits.view(seq * batch, -1).argmax(dim=1)
			total_correct += (preds[mask] == flat_y_types[mask]).sum().item()
			total_count += mask.sum().item()

		total_play_type_loss += play_type_loss.item()
		total_player_loss += player_loss.item()

		if writer is not None:
			writer.add_scalar(
				"step/loss_play_type", play_type_loss.item(), start_step + steps
			)
			writer.add_scalar(
				"step/loss_player", player_loss.item(), start_step + steps
			)
			writer.add_scalar(
				"step/loss_total", loss.item(), start_step + steps
			)
			writer.add_scalar(
				"step/perplexity_play_type",
				play_type_loss.exp().item(),
				start_step + steps,
			)
			writer.add_scalar(
				"step/grad_norm", grad_norm.item(), start_step + steps
			)

		steps += 1

	epoch_play_type_loss = total_play_type_loss / steps
	epoch_player_loss = total_player_loss / steps
	epoch_accuracy = total_correct / total_count if total_count else 0.0

	losses = {
		"play_type": epoch_play_type_loss,
		"player": epoch_player_loss,
		"total": epoch_play_type_loss + epoch_player_loss,
		"play_type_accuracy": epoch_accuracy,
	}

	if writer is not None:
		epoch = start_step // max(steps, 1)
		writer.add_scalar("epoch/loss_play_type", epoch_play_type_loss, epoch)
		writer.add_scalar("epoch/loss_player", epoch_player_loss, epoch)
		writer.add_scalar(
			"epoch/perplexity_play_type",
			torch.tensor(epoch_play_type_loss).exp().item(),
			epoch,
		)
		writer.add_scalar("epoch/play_type_accuracy", epoch_accuracy, epoch)

	return model, losses, steps


def do_validation(
	model, dataset, batch_size, device="cuda:0", writer=None, epoch=0
):
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		collate_fn=collate_fn,
		num_workers=4,
		pin_memory=True,
		persistent_workers=True,
	)

	play_type_loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=0)
	player_loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=0)

	total_play_type_loss = 0.0
	total_player_loss = 0.0
	total_correct = 0
	total_count = 0

	model.eval()
	with torch.no_grad():
		for teams, x_types, x_players, x_state, y_types, y_players in tqdm(
			loader, desc="val", position=1, leave=False
		):
			teams = teams.to(device)
			x_types = x_types.to(device)
			x_players = x_players.to(device)
			x_state = x_state.to(device)
			y_types = y_types.to(device)
			y_players = y_players.to(device)

			play_type_logits, player_logits, _ = model(
				teams, x_types, x_players, x_state, target_play_types=y_types
			)

			seq, batch = y_types.shape
			flat_y_types = y_types.view(seq * batch)
			flat_y_players = y_players.view(seq * batch)

			play_type_loss = play_type_loss_fn(
				play_type_logits.view(seq * batch, -1), flat_y_types
			)
			player_loss = player_loss_fn(
				player_logits.view(seq * batch, -1), flat_y_players
			)

			mask = flat_y_types != 0
			if mask.any():
				preds = play_type_logits.view(seq * batch, -1).argmax(dim=1)
				total_correct += (
					(preds[mask] == flat_y_types[mask]).sum().item()
				)
				total_count += mask.sum().item()

			total_play_type_loss += play_type_loss.item()
			total_player_loss += player_loss.item()

	n = max(1, len(loader))
	losses = {
		"play_type": total_play_type_loss / n,
		"player": total_player_loss / n,
		"total": (total_play_type_loss + total_player_loss) / n,
		"play_type_accuracy": total_correct / total_count
		if total_count
		else 0.0,
	}

	if writer is not None:
		writer.add_scalar("val/loss_play_type", losses["play_type"], epoch)
		writer.add_scalar("val/loss_player", losses["player"], epoch)
		writer.add_scalar("val/loss_total", losses["total"], epoch)
		writer.add_scalar(
			"val/perplexity_play_type",
			torch.tensor(losses["play_type"]).exp().item(),
			epoch,
		)
		writer.add_scalar(
			"val/play_type_accuracy", losses["play_type_accuracy"], epoch
		)

	return losses
