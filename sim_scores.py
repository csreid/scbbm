import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import torch
from score_dataset import ScoreDataset
from gs import GameSimulator
import matplotlib.pyplot as plt

sd = ScoreDataset('cleaned_data.csv')
gs = GameSimulator(len(sd.teams), 69)
try:
	gs.load_state_dict(torch.load('model.ptch'))
except:
	print(f'Failed to load model')

home = 'Purdue'
away = 'Louisville'

scores, times = gs.sim_scores(home, away, sd, length=300)

scores = scores[:, 0]
times = times[:, 0]

plt.plot(times.detach().numpy(), scores[:, 0].detach().numpy(), label=home)
plt.plot(times.detach().numpy(), scores[:, 1].detach().numpy(), label=away)
plt.legend()

plt.show()
