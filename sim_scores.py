import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import torch
from score_dataset import ScoreDataset
from gs import GameSimulator

sd = ScoreDataset('cleaned_data.csv')
gs = GameSimulator(len(sd.teams), 69)
gs.load_state_dict(torch.load('model.ptch'))

scores, times = gs.sim_scores('Purdue', 'Hartford', sd, length=300)

scores = scores[:, 0]
times = times[:, 0]

for (score, time) in zip(scores, times):
	print(f'{int(score[0])} | {int(score[1])} @ {int(time)}')

	if time > 2000:
		break
