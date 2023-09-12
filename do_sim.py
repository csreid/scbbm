import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
import torch

from pbp_dataset import PlayByPlayDataset
from gs import GameSimulator

pbp = PlayByPlayDataset('cleaned_data.csv', min_len=200, max_len=500, size=1)
gs = GameSimulator(len(pbp.teams), len(pbp.plays))##.to('cuda:0')

gs.load_state_dict(torch.load('model.ptch'))
print(f'Successfully loaded model')


plays, times = gs.simulate('Purdue', 'Hartford', pbp, maxlen=100)
for play, time in zip(plays, times):
	print(f'{play} @ {time} seconds')
