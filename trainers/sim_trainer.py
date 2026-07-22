import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from pbp_dataset import collate_fn


def _forward_pass(model, batch, device):
	"""Unpack a collate_fn batch, move to device, run model, return everything."""
	teams, x_types, x_players, x_state, y_types, y_players, player_masks, y_score_delta, y_time_delta_log = batch
	teams          = teams.to(device)
	x_types        = x_types.to(device)
	x_players      = x_players.to(device)
	x_state        = x_state.to(device)
	y_types        = y_types.to(device)
	y_players      = y_players.to(device)
	player_masks   = player_masks.to(device)
	y_score_delta  = y_score_delta.to(device)
	y_time_delta_log = y_time_delta_log.to(device)

	play_type_logits, player_logits, score_delta_logits, time_delta_pred, _ = model(
		teams, x_types, x_players, x_state,
		target_play_types=y_types,
		player_mask=player_masks,
	)
	return (
		play_type_logits, player_logits, score_delta_logits, time_delta_pred,
		y_types, y_players, y_score_delta, y_time_delta_log,
	)


def _compute_losses(play_type_logits, player_logits, score_delta_logits, time_delta_pred,
                    y_types, y_players, y_score_delta, y_time_delta_log,
                    play_type_loss_fn, player_loss_fn):
	seq, batch = y_types.shape
	type_mask = y_types.view(seq * batch) != 0   # non-PAD positions

	play_type_loss = play_type_loss_fn(
		play_type_logits.view(seq * batch, -1),
		y_types.view(seq * batch),
	)
	player_loss = player_loss_fn(
		player_logits.view(seq * batch, -1),
		y_players.view(seq * batch),
	)

	# Score delta: CrossEntropy over 7 classes, ignoring PAD positions
	flat_sd_logits = score_delta_logits.view(seq * batch, -1)
	flat_y_sd = y_score_delta.view(seq * batch)
	score_delta_loss = (
		F.cross_entropy(flat_sd_logits[type_mask], flat_y_sd[type_mask])
		if type_mask.any() else play_type_loss.new_tensor(0.0)
	)

	# Time delta: Huber loss in log1p-seconds space, ignoring PAD positions
	flat_td_pred = time_delta_pred.squeeze(-1).view(seq * batch)
	flat_y_td = y_time_delta_log.view(seq * batch)
	time_delta_loss = (
		F.huber_loss(flat_td_pred[type_mask], flat_y_td[type_mask])
		if type_mask.any() else play_type_loss.new_tensor(0.0)
	)

	total = play_type_loss + player_loss + score_delta_loss + time_delta_loss
	return play_type_loss, player_loss, score_delta_loss, time_delta_loss, total


def _compute_accs(play_type_logits, player_logits, y_types, y_players):
	seq, batch = y_types.shape
	flat_y_types   = y_types.view(seq * batch)
	flat_y_players = y_players.view(seq * batch)

	type_mask   = flat_y_types != 0
	player_mask = flat_y_players != 0

	type_acc = player_acc = 0.0
	if type_mask.any():
		preds = play_type_logits.view(seq * batch, -1).argmax(dim=1)
		type_acc = (preds[type_mask] == flat_y_types[type_mask]).float().mean().item()
	if player_mask.any():
		preds = player_logits.view(seq * batch, -1).argmax(dim=1)
		player_acc = (preds[player_mask] == flat_y_players[player_mask]).float().mean().item()
	return type_acc, player_acc


def _player_entropy(player_logits, y_players):
	seq, batch = y_players.shape
	mask = y_players.view(seq * batch) != 0
	if not mask.any():
		return 0.0
	flat = player_logits.view(seq * batch, -1)
	probs = F.softmax(flat[mask], dim=-1)
	return (-(probs * probs.clamp(min=1e-9).log()).sum(dim=-1)).mean().item()


def do_epoch(model, dataset, batch_size, optimizer, device="cuda:0",
             save_to=None, writer=None, start_step=0):
	loader = DataLoader(
		dataset, batch_size=batch_size, collate_fn=collate_fn,
		shuffle=True, num_workers=4, pin_memory=True,
	)

	play_type_loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=0)
	player_loss_fn    = CrossEntropyLoss(reduction="mean", ignore_index=0)

	totals = dict(play_type=0., player=0., score_delta=0., time_delta=0.,
	              total=0., type_acc=0., player_acc=0., entropy=0.)
	steps = 0
	model.train()

	for _, batch in tqdm(enumerate(loader), total=len(dataset) // batch_size,
	                     position=1, leave=False):
		outputs = _forward_pass(model, batch, device)
		play_type_logits, player_logits, score_delta_logits, time_delta_pred, \
			y_types, y_players, y_score_delta, y_time_delta_log = outputs

		pt_loss, pl_loss, sd_loss, td_loss, loss = _compute_losses(
			play_type_logits, player_logits, score_delta_logits, time_delta_pred,
			y_types, y_players, y_score_delta, y_time_delta_log,
			play_type_loss_fn, player_loss_fn,
		)

		optimizer.zero_grad()
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
		optimizer.step()

		type_acc, player_acc = _compute_accs(play_type_logits, player_logits, y_types, y_players)
		entropy = _player_entropy(player_logits, y_players)

		totals["play_type"]   += pt_loss.item()
		totals["player"]      += pl_loss.item()
		totals["score_delta"] += sd_loss.item()
		totals["time_delta"]  += td_loss.item()
		totals["total"]       += loss.item()
		totals["type_acc"]    += type_acc
		totals["player_acc"]  += player_acc
		totals["entropy"]     += entropy

		if writer is not None:
			s = start_step + steps
			writer.add_scalar("step/loss_play_type",         pt_loss.item(),          s)
			writer.add_scalar("step/loss_player",            pl_loss.item(),          s)
			writer.add_scalar("step/loss_score_delta",       sd_loss.item(),          s)
			writer.add_scalar("step/loss_time_delta",        td_loss.item(),          s)
			writer.add_scalar("step/loss_total",             loss.item(),             s)
			writer.add_scalar("step/perplexity_play_type",   pt_loss.exp().item(),    s)
			writer.add_scalar("step/grad_norm",              grad_norm.item(),        s)
		steps += 1

	n = max(1, steps)
	losses = {
		"play_type":          totals["play_type"]   / n,
		"player":             totals["player"]      / n,
		"score_delta":        totals["score_delta"] / n,
		"time_delta":         totals["time_delta"]  / n,
		"total":              totals["total"]        / n,
		"play_type_accuracy": totals["type_acc"]    / n,
		"player_accuracy":    totals["player_acc"]  / n,
		"player_entropy":     totals["entropy"]     / n,
	}

	if writer is not None:
		epoch = start_step // n
		writer.add_scalar("epoch/loss_play_type",         losses["play_type"],          epoch)
		writer.add_scalar("epoch/loss_player",            losses["player"],             epoch)
		writer.add_scalar("epoch/loss_score_delta",       losses["score_delta"],        epoch)
		writer.add_scalar("epoch/loss_time_delta",        losses["time_delta"],         epoch)
		writer.add_scalar("epoch/perplexity_play_type",   torch.tensor(losses["play_type"]).exp().item(),   epoch)
		writer.add_scalar("epoch/perplexity_player",      torch.tensor(losses["player"]).exp().item(),      epoch)
		writer.add_scalar("epoch/play_type_accuracy",     losses["play_type_accuracy"], epoch)
		writer.add_scalar("epoch/player_accuracy",        losses["player_accuracy"],    epoch)
		writer.add_scalar("epoch/player_entropy",         losses["player_entropy"],     epoch)

	return model, losses, steps


def do_validation(model, dataset, batch_size, device="cuda:0", writer=None, epoch=0):
	loader = DataLoader(
		dataset, batch_size=batch_size, collate_fn=collate_fn,
		num_workers=4, pin_memory=True, persistent_workers=True,
	)

	play_type_loss_fn = CrossEntropyLoss(reduction="mean", ignore_index=0)
	player_loss_fn    = CrossEntropyLoss(reduction="mean", ignore_index=0)

	totals = dict(play_type=0., player=0., score_delta=0., time_delta=0.,
	              total=0., type_acc=0., player_acc=0., entropy=0.)
	steps = 0
	model.eval()

	with torch.no_grad():
		for batch in tqdm(loader, desc="val", position=1, leave=False):
			outputs = _forward_pass(model, batch, device)
			play_type_logits, player_logits, score_delta_logits, time_delta_pred, \
				y_types, y_players, y_score_delta, y_time_delta_log = outputs

			pt_loss, pl_loss, sd_loss, td_loss, loss = _compute_losses(
				play_type_logits, player_logits, score_delta_logits, time_delta_pred,
				y_types, y_players, y_score_delta, y_time_delta_log,
				play_type_loss_fn, player_loss_fn,
			)

			type_acc, player_acc = _compute_accs(play_type_logits, player_logits, y_types, y_players)
			entropy = _player_entropy(player_logits, y_players)

			totals["play_type"]   += pt_loss.item()
			totals["player"]      += pl_loss.item()
			totals["score_delta"] += sd_loss.item()
			totals["time_delta"]  += td_loss.item()
			totals["total"]       += loss.item()
			totals["type_acc"]    += type_acc
			totals["player_acc"]  += player_acc
			totals["entropy"]     += entropy
			steps += 1

	n = max(1, steps)
	losses = {
		"play_type":          totals["play_type"]   / n,
		"player":             totals["player"]      / n,
		"score_delta":        totals["score_delta"] / n,
		"time_delta":         totals["time_delta"]  / n,
		"total":              totals["total"]        / n,
		"play_type_accuracy": totals["type_acc"]    / n,
		"player_accuracy":    totals["player_acc"]  / n,
		"player_entropy":     totals["entropy"]     / n,
	}

	if writer is not None:
		writer.add_scalar("val/loss_play_type",         losses["play_type"],          epoch)
		writer.add_scalar("val/loss_player",            losses["player"],             epoch)
		writer.add_scalar("val/loss_score_delta",       losses["score_delta"],        epoch)
		writer.add_scalar("val/loss_time_delta",        losses["time_delta"],         epoch)
		writer.add_scalar("val/loss_total",             losses["total"],              epoch)
		writer.add_scalar("val/perplexity_play_type",   torch.tensor(losses["play_type"]).exp().item(),  epoch)
		writer.add_scalar("val/perplexity_player",      torch.tensor(losses["player"]).exp().item(),     epoch)
		writer.add_scalar("val/play_type_accuracy",     losses["play_type_accuracy"], epoch)
		writer.add_scalar("val/player_accuracy",        losses["player_accuracy"],    epoch)
		writer.add_scalar("val/player_entropy",         losses["player_entropy"],     epoch)

	return losses
