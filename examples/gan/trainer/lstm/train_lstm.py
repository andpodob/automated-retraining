import argparse
import io
import os
import random
import sys

import PIL
from matplotlib import pyplot as plt
import numpy as np
import torch

# Add parent directory to the system path to access utils.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch.utils.tensorboard import SummaryWriter
import lstm_model
from utils import set_log_dir
from torchvision.transforms import ToTensor
from trainer import DataSet

_EXP_NAME = 'BTC_LSTM'
_DEVICE = 'cuda'

_LEARNING_RATE = 1e-4
_WEIGHT_DECAY = 1e-7

def main():
    num_cores = os.cpu_count() 
    torch.set_num_threads(num_cores) 
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default=None, type=int,
                    help='experiment id')
    parser.add_argument('--exp_name', default=None, type=str,
                    help='experiment name')
    parser.add_argument('--val_set_ratio', default=None, type=float,
                    help='val_set_ratio')
    parser.add_argument('--seq_len', default=None, type=int,
                    help='seq_len')
    parser.add_argument('--prediction_size', default=10, type=int,
                    help='prediction_size')
    parser.add_argument('--augmentation', default=0, type=int,
                        help='augmentation factor')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs')
    parser.add_argument('--gen_checkpoint', default=None, type=str,
                        help='checkpoint file for generator')
    parser.add_argument('--load_file', default=None, type=str,
                    help='checkpoint path to load checkpoint')
    parser.add_argument('--training_set_path', default=None, type=str,
                    help='path to training set')
    parser.add_argument('--test_set_path', default=None, type=str,
                    help='path to test set')
    parser.add_argument('--logs_dir', default=None, type=str,
                    help='directory for logs')
    parser.add_argument('--checkpoint_file', default=None, type=str,
                    help='checkpoint path to save checkpoint')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training.')

    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    lstm = lstm_model.LSTMModel(input_dim=1, output_dim=args.prediction_size, layer_dim=3, hidden_dim=64, dropout_prob=0.2)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    lstm.to(_DEVICE)
    writer = None
    path_helper = None
    start_epoch = 0
    if args.load_file:
        checkpoint = torch.load(args.load_file, map_location=_DEVICE)
        lstm.load_state_dict(checkpoint["lstm_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        path_helper = checkpoint["path_helper"]
        writer = SummaryWriter(path_helper['log_path'])
        start_epoch = checkpoint['epoch']
    else:
        path_helper = set_log_dir(args.logs_dir, args.exp_name)
        writer = SummaryWriter(path_helper['log_path'])

    train_set, val_set = torch.utils.data.random_split(DataSet(torch.load(args.training_set_path), args.seq_len - args.prediction_size, args.seq_len), [1 - args.val_set_ratio, args.val_set_ratio]) 
    train_loader = torch.utils.data.DataLoader(train_set,  batch_size=16, num_workers=8, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, num_workers=8, shuffle=True)

    test_set = DataSet(torch.load(args.test_set_path), args.seq_len - args.prediction_size, args.seq_len)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, num_workers=8, shuffle=True)

    writer_dict = {
        'writer': writer,
        'train_steps': start_epoch * len(train_loader),
    }

    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimization = Optimization(lstm, loss_fn, optimizer, seq_len=args.seq_len, prediction_size=args.prediction_size)

    optimization.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        writer_dict=writer_dict,
        start_epoch=start_epoch,
        path_helper=path_helper, 
        batch_size=16,
        n_epochs=args.epochs
    )
    evaluation_loss = optimization.evaluate(test_loader, 16, 1)
    writer.add_scalar('evaluation_loss', evaluation_loss, 0)



class Optimization:
    def __init__(self, model, loss_fn, optimizer, seq_len, prediction_size):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.seq_len = seq_len
        self.prediction_size = prediction_size
        
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()
        # Makes predictions
        yhat = self.model(x)
        # Computes loss
        y = y.squeeze().reshape(yhat.shape[0], yhat.shape[1])
        loss = self.loss_fn(y, yhat)
        if loss.item() > 10:
            return None
        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, 
              train_loader, 
              val_loader,
              test_loader, 
              writer_dict, 
              start_epoch, 
              path_helper, 
              batch_size=16, 
              n_epochs=100):
        writer = writer_dict['writer']
        for epoch in range(start_epoch, n_epochs + 1):
            batch_losses = []
            for seq, _ in train_loader:
                [x_batch, y_batch] = torch.split(seq, [self.seq_len - self.prediction_size, self.prediction_size], dim=3)
                steps = writer_dict['train_steps']
                if len(x_batch) != batch_size:
                    continue
                x_batch = x_batch.view([batch_size, -1, 1]).to(_DEVICE)
                y_batch =  y_batch.to(_DEVICE)
                loss = self.train_step(x_batch, y_batch)
                if loss == None:
                    continue
                batch_losses.append(loss)
                writer.add_scalar('loss', loss, steps)
                writer_dict['train_steps'] = steps + 1
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for seq, _ in val_loader:
                    [x_val, y_val] = torch.split(seq, [self.seq_len - self.prediction_size, self.prediction_size], dim=3)
                    if len(x_val) != batch_size:
                        continue
                    x_val = x_val.view([batch_size, -1, 1]).to(_DEVICE)
                    y_val = y_val.to(_DEVICE)
                    self.model.eval()
                    yhat = self.model(x_val)
                    y_val = y_val.squeeze().reshape(yhat.shape[0], yhat.shape[1])
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.8f}\t Validation loss: {validation_loss:.8f}"
                )
                plot_buf = self.gen_plot(test_loader, epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).unsqueeze(0)
                writer.add_image('Image', image[0], epoch)

            torch.save({
                'epoch': epoch,
                'lstm_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'path_helper': path_helper
            }, os.path.join(path_helper['ckpt_path'], 'checkpoint'))

    def gen_plot(self, test_loader, epoch):
        N = 16
        self.model.eval()
        fig, axs = plt.subplots(N, 1, figsize=(8,8))
        fig.suptitle(f'Forecasting data at epoch {epoch}', fontsize=30)
        seq_batch = next(iter(test_loader)) 
        for seq, _ in test_loader:
            [X, Y] = torch.split(seq, [self.seq_len - self.prediction_size, self.prediction_size], dim=3)  
            x_val = X.view([N, -1, 1]).to(_DEVICE)
            yhat = self.model(x_val).detach().cpu()
            for i in range(N):
                y_pred = np.concatenate([X[i][0][0], yhat[i]])
                y_org = np.concatenate([X[i][0][0], Y[i][0][0]])
                axs[i].plot([i for i in range(len(y_pred))], y_pred)
                axs[i].plot([i for i in range(len(y_org))], y_org)
            break
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        return buf

    def evaluate(self, test_loader, batch_size=16, n_features=1):        
        with torch.no_grad():
            losses = []
            for seq, _ in test_loader:
                [x_test, y_test] = torch.split(seq, [self.seq_len - self.prediction_size, self.prediction_size], dim=3)
                x_test = x_test.view([batch_size, -1, n_features]).to(_DEVICE)
                y_test = y_test.to(_DEVICE)
                self.model.eval()
                yhat = self.model(x_test)
                y_test = y_test.squeeze().reshape(yhat.shape[0], yhat.shape[1])
                test_loss = self.loss_fn(y_test, yhat).item()
                losses.append(test_loss)

        return np.mean(losses)

    def plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.style.use('ggplot')
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()



if __name__ == "__main__":
    main()
