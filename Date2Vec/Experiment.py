from Model import Date2Vec
from Data import TimeDateDataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

class Date2VecExperiment:
    def __init__(self, model, act, optim='adam', lr=0.001, batch_size=256, num_epoch=50, cuda=False):
        self.model = model
        if cuda:
            self.model = model.cuda()
        self.optim = optim
        self.lr = lr
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.cuda = cuda
        self.act = act
        
        with open('date_time.txt', 'r') as f:
            full = f.readlines()
        train = full[len(full)//3: 2*len(full)//3]
        test_prev = full[:len(full)//3]
        test_after = full[2*len(full)//3:]

        self.train_dataset = TimeDateDataset(train)
        self.test_prev_dataset = TimeDateDataset(test_prev)
        self.test_after_dataset = TimeDateDataset(test_after)
        
    def train(self):
        #loss_fn1 = torch.nn.L1Loss()
        loss_fn = torch.nn.MSELoss()
        #loss_fn = lambda y_true, y_pred: loss_fn1(y_true, y_pred) + loss_fn2(y_true, y_pred)

        if self.cuda:
            loss_fn = loss_fn.cuda()
            self.model = self.model.cuda()

        if self.optim == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == 'sgd_momentum':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=self.cuda)
        test1_dataloader = DataLoader(self.test_prev_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=self.cuda)
        test2_dataloader = DataLoader(self.test_after_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=self.cuda)
                
        
        avg_best = 1000000000000000
        best_weights = self.model.state_dict()
        avg_loss = 0
        for ep in range(self.num_epoch):
            train_loss = 0.0
            for (x, y), (x_prev, y_prev), (x_after, y_after) in zip(train_dataloader, test1_dataloader, test2_dataloader):
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    x_prev = x_prev.cuda()
                    y_prev = y_prev.cuda()
                    x_after = x_after.cuda()
                    y_after = y_after.cuda()
                optimizer.zero_grad()
                y_pred = self.model(x)      
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                with torch.no_grad():
                    y_pred_prev = self.model(x_prev)
                    r2_prev = r2_score(y_prev.cpu().numpy(), y_pred_prev.cpu().numpy())
                    mae_prev = mean_absolute_error(y_prev.cpu().numpy(), y_pred_prev.cpu().numpy())
                    mse_prev = mean_squared_error(y_prev.cpu().numpy(), y_pred_prev.cpu().numpy())

                    y_pred_after = self.model(x_after)
                    r2_after = r2_score(y_after.cpu().numpy(), y_pred_after.cpu().numpy())
                    mae_after = mean_absolute_error(y_after.cpu().numpy(), y_pred_after.cpu().numpy())
                    mse_after = mean_squared_error(y_after.cpu().numpy(), y_pred_after.cpu().numpy())
                    if avg_loss == 0:
                        avg_loss = (mae_prev+mse_prev+mae_after+mse_after)/4
                    avg_loss = (loss.item() + avg_loss) / 2
                if avg_loss < avg_best:
                    avg_best = avg_loss
                    best_weights = self.model.state_dict()
            print('Epoch: {}/{}, Training loss: {}, Avg loss: {}, Best avg loss: {}'.format(ep, self.num_epoch, train_loss, avg_loss, avg_best))

        self.model.load_state_dict(best_weights)
        torch.save(self.model, "./models/d2v_{}_{}.pth".format(self.act, avg_best))
    
    def test(self):
        test1_dataloader = DataLoader(self.test_prev_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=self.cuda)
        test2_dataloader = DataLoader(self.test_after_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=self.cuda)

        to_int = lambda dt: list(map(int, dt))

        total_pred_test1 = len(self.test_prev_dataset)
        total_pred_test2 = len(self.test_after_dataset)
        
        correct_pred_prev = 0
        correct_pred_after = 0

        def count_correct(ypred, ytrue):
            c = 0
            for p, t in zip(ypred, ytrue):
                for pi, ti in zip(to_int(p), to_int(t)):
                    if pi == ti:
                        c += 1
            return c

        for (x_prev, y_prev), (x_after, y_after) in zip(test1_dataloader, test2_dataloader):
            with torch.no_grad():
                y_pred_prev = self.model(x_prev).cpu().numpy().tolist()
                correct_pred_prev += count_correct(y_pred_prev, y_prev.cpu().numpy().tolist())

                y_pred_after = self.model(x_after)
                correct_pred_after += count_correct(y_pred_after, y_after.cpu().numpy().tolist())
        
        prev_acc = correct_pred_prev / total_pred_test1
        after_acc = correct_pred_after / total_pred_test2

        return prev_acc, after_acc        
if __name__ == "__main__":
    act = 'cos'
    optim = 'adam'
    os.system("mkdir ./models/")

    m = Date2Vec(k=64, act=act)
    #m = torch.load("models/sin/nextdate_11147_23.02417500813802.pth")
    cuda = torch.cuda.is_available()
    if cuda:
        print('GPU available!')
    exp = Date2VecExperiment(m, act, lr=0.001, num_epoch=100, batch_size=512, cuda=cuda, optim=optim)
    exp.train()
    #test1_acc, test2_acc = exp.test()
    #print("test1 accuracy:{}, test2 accuracy:{}".format(test1_acc, test2_acc))
