from tqdm import trange
import torch, copy, json, os, numpy as np
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import ExponentialLR


class TrainDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, additional_data, embedding_model):
        self.data = data
        self.additional_data = additional_data
        self.model = embedding_model
        dates = [date.split(",") for date in data['datetime']]
        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datetime = self.dates[idx]
        target = self.data.iloc[idx].values[1:].astype(float)
        features = self.model.encode(torch.FloatTensor(datetime)).detach()
        add_features = torch.FloatTensor(self.additional_data.iloc[idx].values[:4])
        features = torch.cat([features, add_features]).unsqueeze(0)
        target = torch.FloatTensor(target)
        return features, target
    
class TestDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, additional_data, embedding_model):
        self.data = data
        self.additional_data = additional_data
        self.model = embedding_model
        dates = [date.split(",") for date in data['datetime']]
        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datetime = self.dates[idx]
        features = self.model.encode(torch.FloatTensor(datetime)).detach()
        add_features = torch.FloatTensor(self.additional_data.iloc[idx].values[:4])
        features = torch.cat([features, add_features]).unsqueeze(0)
        return features
    
def train(model, train_dataloader, epochs, valid_dataloader=None, fold=0, decay_rate=0.0, lr=0.001, save_model=True, save_every=10, optimizer='Adam'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu": print("\nTraining on CPU, it may take long...")
    else: print("\nGPU available !")
    print()
    print("#"*50)
    print()
    print("{} starts training... \n".format(model.name))
    print("#"*50, "\n")
    if device.type == "cuda": model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if decay_rate: scheduler = ExponentialLR(opt, decay_rate)
    Train_loss = []; Valid_loss = []; best_score = np.inf
    Epochs = trange(epochs, desc=f'Fold: {fold}, Previous Loss: {np.nan}, Current Loss: {np.nan}, Best Loss: {np.nan}', leave=True)
    best_epoch = np.nan
    for e in Epochs:
        train_losses = []
        for x, target in train_dataloader:
            if device.type == "cuda":
                x, target = x.cuda(), target.cuda()
            scores = model(x)
            loss = model.loss(scores, target)
            train_losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            clip_grad_value_(model.parameters(), clip_value=5.0)
            opt.step()
            if decay_rate: scheduler.step()
        Train_loss.append(np.mean(train_losses))
        if valid_dataloader is not None:
            model.eval()
            with torch.no_grad():
                valid_losses = []
                for xx, targett in valid_dataloader:
                    if device.type == "cuda":
                        xx, targett = xx.cuda(), targett.cuda()
                    scores_ = model(xx)
                    loss_ = model.loss(scores_, targett)
                    valid_losses.append(loss_.item())
            Valid_loss.append(np.mean(valid_losses))
        if len(Train_loss) >= 2:
            if valid_dataloader is not None:
                Epochs.set_description('Fold: {}, Previous Loss: {:.4f}, Current Loss: {:.4f}, Best Loss: {:.4f}, Validation Loss: {:.4f}'.format(fold, Train_loss[-2], Train_loss[-1], min(Train_loss), Valid_loss[-1]))
            else:
                Epochs.set_description('Fold: {}, Previous Loss: {:.4f}, Current Loss: {:.4f}, Best Loss: {:.4f}'.format(fold, Train_loss[-2], Train_loss[-1], min(Train_loss)))
        else:
            Epochs.set_description('Fold: {}, Previous Loss: {:.4f}, Current Loss: {:.4f}, Best Loss: {:.4f}'.format(fold, np.nan, Train_loss[-1], min(Train_loss)))
        Epochs.refresh()
        model.train()
        weights = copy.deepcopy(model.state_dict())
        if valid_dataloader is not None:
            if Valid_loss and Valid_loss[-1] < best_score:
                best_score = Valid_loss[-1]
                best_weights = weights
                best_epoch = e
        else:
            if Train_loss and Train_loss[-1] < best_score:
                best_score = Train_loss[-1]
                best_weights = weights
                best_epoch = e
        """
        if e > 0 and e%save_every == 0:
            path = "./trained_models/"+"trained_"+model.name+f"_proj_dim{model.proj_dim}_snapshot{e}_fold{fold}.pt"
            if model.name == 'SetTransformer':
                path = "./trained_models/"+"trained_"+model.name+f"_proj_dim{model.proj_dim}_num_inds"+str(model.num_inds)+"_num_heads"+str(model.num_heads)+f"_snapshot{e}_fold{fold}.pt"
            torch.save(weights, path)
            print(f'Saved snapshot {e}')
        """
    #model.load_state_dict(best_weights)
    print("\nBest loss on validation: ", min(Valid_loss))
    print()
    if save_model:
        if not os.path.exists("./trained_models/"):
            os.mkdir("./trained_models/")
        #path = "./trained_models/"+"trained_"+model.name+f"_proj_dim{model.proj_dim}_fold{fold}.pt"
        path = "./trained_models/"+"trained_"+model.name+f"_fold{fold}.pt"
        if model.name == 'SetTransformer':
            path = "./trained_models/"+"trained_"+model.name+f"_proj_dim{model.proj_dim}_num_inds"+str(model.num_inds)+"_num_heads"+str(model.num_heads)+f"_fold{fold}.pt"
        torch.save(best_weights, path)
        print("{} saved".format(model.name))
    #if not os.path.exists("./metrics/"):
    #    os.mkdir("./metrics/")
    #path = "metrics_"+model.name+f"_proj_dim{model.proj_dim}_fold{fold}.json" if model.name != 'SetTransformer' else "metrics_"+model.name+f"_proj_dim{model.proj_dim}_num_inds"+str(model.num_inds)+"_num_heads"+str(model.num_heads)+f"_fold{fold}.json"
    #with open("./metrics/"+path, "w") as results_file:
    #    json.dump({"loss": Train_loss, "best epoch": best_epoch, "best loss": best_score}, results_file, indent=3)