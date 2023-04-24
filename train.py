from utils import TrainDataLoader, train
import pandas as pd, numpy as np
import torch, random, os
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from models import *
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')
        
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='+', default=['SetTransformer', 'GRU', 'LSTM', 'MLP'], help='Neural models')
parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers to use to load training data')
parser.add_argument('--input_dim', type=int, default=68, help='The projection dimension for examples')
parser.add_argument('--proj_dim', type=int, default=64, help='The projection dimension for examples')
parser.add_argument('--decay_rate', type=float, default=0.0, help='Decay rate for the optimizer')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--num_inds', type=int, nargs='+', default=[4, 8, 16, 32], help='Number of induced points')
parser.add_argument('--num_heads', type=int, nargs='+', default=[2, 4], help='Number of attention heads')
parser.add_argument('--n_layers', type=int, default=2, help='Number of recurrent network layers')
parser.add_argument('--save_every', type=int, default=10, help='Save model weights at the given snapshot')
parser.add_argument('--save_model', type=str2bool, default=True, help='Whether to save the model after training')
args = parser.parse_args()
embedding_model = torch.load('Date2Vec/models/d2v_cos_14.054091384440834.pth', map_location='cpu').eval()    
df_train = pd.read_csv('data_clean.csv')
additional_data = pd.read_csv('data_train_additional.csv')
valid_dataloader = None


Kf = KFold(n_splits=args.folds, shuffle=True, random_state=142)
for fold, (train_idx, valid_idx) in enumerate(Kf.split(np.arange(len(df_train)))):
    print('Train indices: ', train_idx)
    print('Validation indices: ', valid_idx)
    print('Validation data size: ', len(valid_idx))
    print('Overlap: ', set(train_idx).intersection(set(valid_idx)))
    train_dataset = TrainDataLoader(df_train.iloc[train_idx], additional_data.iloc[train_idx], embedding_model)
    valid_dataset = TrainDataLoader(df_train.iloc[valid_idx], additional_data.iloc[valid_idx], embedding_model)
    train_dataloader = DataLoader(train_dataset, batch_size=256, num_workers=8, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=256, num_workers=8, shuffle=False)
    for m in args.models:
        if m == 'SetTransformer':
            for num_inds in args.num_inds:
                for num_heads in args.num_heads:
                    print("Num_inds: ", num_inds, "Num_heads: ", num_heads)
                    model = SetTransformer(args.input_dim, args.proj_dim, num_inds, num_heads, 1)
                    train(model, train_dataloader, epochs=args.epochs, valid_dataloader=valid_dataloader, fold=fold, decay_rate=args.decay_rate, lr=args.lr, save_model=args.save_model, optimizer='Adam')
        else:
            if m == 'LSTM':
                model = LSTM(args.input_dim, args.proj_dim, args.n_layers)
            elif m == 'GRU':
                model = GRU(args.input_dim, args.proj_dim, args.n_layers)
            elif m == 'MLP':
                model = MLP(args.input_dim, args.proj_dim)
            train(model, train_dataloader, epochs=args.epochs, valid_dataloader=valid_dataloader, fold=fold, decay_rate=args.decay_rate, lr=args.lr, save_model=args.save_model, save_every=args.save_every, optimizer='Adam')

