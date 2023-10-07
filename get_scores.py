import sys
sys.path.append('/home/csreid/.pyenv/versions/3.10.12/lib/python3.10/site-packages')
from pbp_dataset import PlayByPlayDataset
from score_dataset import ScoreDataset
from final_score_dataset import FinalScoreDataset
import torch
import pandas as pd
from torch.nn import GRU, CrossEntropyLoss, Module, KLDivLoss, LSTM, Linear, Embedding, functional as F, RNN, MSELoss, BCEWithLogitsLoss, L1Loss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import embed
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gs import GameSimulator

fsd = FinalScoreDataset('cleaned_data.csv')
gs = GameSimulator(len(fsd.teams), 69)
gs.load_state_dict(torch.load('model.ptch'))

teams = fsd._teams_to_tensor('Purdue', 'Gonzaga').long().unsqueeze(0)

print(gs.final_score(teams))
