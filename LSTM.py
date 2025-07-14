# Gideon Vos 2025
# James Cook University, Australia
# www.linkedin/in/gideonvos

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import random
import scipy.stats as stats

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]
        if len(x.shape) > 2:
            new_shape = (x.shape[0], -1)
            x = x.reshape(new_shape)
        elif len(x.shape) == 1:
            x = x.reshape(self.seq_length, 1)
        
        y = self.targets[idx+self.seq_length]
        return x, y


class LSTM_Forecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.0, use_mc_dropout=False, use_synaptic_pruning=False):
        super(LSTM_Forecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_mc_dropout = use_mc_dropout
        self.use_synaptic_pruning = use_synaptic_pruning
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)
        self.synaptic_pruning = None
        if self.use_synaptic_pruning:
            self.synaptic_pruning = SynapticPruning(
                modules=[self.lstm, self.fc],
                # Set your hyperparameters here below:
                max_prune_rate=1.2,
                min_prune_rate=0.7,
                warmup_epochs=2,
                prune_every=2,
                total_epochs=20
            )

    def forward(self, x, target=None, epoch=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) > 3:
            x = x.view(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        if self.use_mc_dropout:
            lstm_out = self.dropout(lstm_out)
        elif self.dropout.p > 0 and self.training:
            lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output


class SynapticPruning:
    def __init__(self, modules, max_prune_rate, min_prune_rate, warmup_epochs, prune_every, total_epochs):
        self.modules = modules
        self.max_prune_rate = max_prune_rate
        self.min_prune_rate = min_prune_rate
        self.warmup_epochs = warmup_epochs
        self.prune_every = prune_every
        self.total_epochs = total_epochs
        self.batches_processed = 0
        self.epoch = 0
        self.device = None
        self.masks = {}
        self._initialize_pruning_masks()

    def _initialize_pruning_masks(self):
        for module in self.modules:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    self.masks[param] = torch.ones_like(param.data, device='cpu')

    def set_device(self, device):
        self.device = device
        for param, mask in self.masks.items():
            self.masks[param] = mask.to(device)

    def _calculate_target_sparsity(self):
        if self.epoch < self.warmup_epochs:
            return 0.0

        progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        target_sparsity = self.min_prune_rate + (self.max_prune_rate - self.min_prune_rate) * (progress ** 3)
        return target_sparsity

    def _update_masks(self):
        target_sparsity = self._calculate_target_sparsity()
        if target_sparsity <= 0:
            return

        all_weights = []
        param_info = []
        
        for module in self.modules:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    current_mask = self.masks[param]
                    active_indices = current_mask.bool()
                    active_weights = param.data[active_indices]
                    if active_weights.numel() > 0:
                        all_weights.append(torch.abs(active_weights))
                        param_info.append((param, active_indices))
        if not all_weights:
            return
            
        all_weights_tensor = torch.cat(all_weights)
        total_weights = sum(param.numel() for param, _ in param_info)
        current_active_weights = all_weights_tensor.numel()
        target_total_pruned = int(target_sparsity * total_weights)
        currently_pruned = total_weights - current_active_weights
        additional_to_prune = max(0, target_total_pruned - currently_pruned)
        if additional_to_prune == 0 or additional_to_prune >= current_active_weights:
            return

        threshold, _ = torch.kthvalue(all_weights_tensor, additional_to_prune)
        for param, active_indices in param_info:
            active_weights = param.data[active_indices]
            weights_to_prune = torch.abs(active_weights) < threshold
            new_mask = self.masks[param].clone()
            active_positions = torch.where(active_indices)
            for i, should_prune in enumerate(weights_to_prune):
                if should_prune:
                    pos = tuple(coord[i] for coord in active_positions)
                    new_mask[pos] = 0.0
            
            self.masks[param] = new_mask

    def _apply_pruning_masks(self):
        for module in self.modules:
            for name, param in module.named_parameters():
                if 'weight' in name and param in self.masks:
                    mask = self.masks[param]
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                        self.masks[param] = mask
                    param.data *= mask

    def update_pruning(self, epoch):
        self.epoch = epoch
        self.batches_processed += 1
        should_prune = (
            self.epoch >= self.warmup_epochs and
            self.batches_processed % self.prune_every == 0
        )

        if should_prune:
            self._update_masks()
            self._apply_pruning_masks()
            return True
        return False
        
    def get_sparsity_stats(self):
        stats = {}
        layer_idx = 0
        for module in self.modules:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    mask = self.masks[param]
                    total_weights = mask.numel()
                    pruned_weights = (mask == 0).sum().item()
                    sparsity = pruned_weights / total_weights
                    stats[f"{module.__class__.__name__}_{name}_{layer_idx}"] = {
                        'total_weights': total_weights,
                        'pruned_weights': pruned_weights,
                        'sparsity': sparsity
                    }
            layer_idx += 1
        return stats


def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    for i in range(len(X) - seq_length):
        seq = X[i:i+seq_length]
        if len(seq.shape) > 2:
            new_shape = (seq.shape[0], -1)
            seq = seq.reshape(new_shape)
        X_seq.append(seq)
        y_seq.append(y[i+seq_length])

    result_X = np.array(X_seq)
    result_y = np.array(y_seq)
    
    if len(result_X.shape) > 3:
        result_X = result_X.reshape(result_X.shape[0], result_X.shape[1], -1)
        
    return result_X, result_y


def train_test_split_ts(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def early_stopping(val_losses, patience=5):
    if len(val_losses) < patience + 1:
        return False
    for i in range(patience):
        if val_losses[-(i+1)] < val_losses[-(i+2)]:
            return False
    return True


def train_model(model, train_loader, val_loader, epochs=20, patience=3, device='cuda'):
    model.to(device)
    if model.use_synaptic_pruning and model.synaptic_pruning:
        model.synaptic_pruning.set_device(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, y_batch, epoch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()

            if model.use_synaptic_pruning and model.synaptic_pruning:
                model.synaptic_pruning.update_pruning(epoch)
            
            optimizer.step()

            if model.use_synaptic_pruning and model.synaptic_pruning:
                 model.synaptic_pruning._apply_pruning_masks()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                if model.use_mc_dropout:
                    model.train()
                val_outputs = model(X_val, y_val, epoch)
                val_loss += criterion(val_outputs, y_val.unsqueeze(1)).item() * X_val.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if early_stopping(val_losses, patience):
            break

    training_time = time.time() - start_time
    return model, train_losses, val_losses, best_val_loss, epoch + 1, training_time


def evaluate_model(model, test_loader, scaler_y, device='cuda'):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            if model.use_mc_dropout:
                model.train()
            outputs = model(X_test)
            
            y_true.extend(y_test.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy().flatten())

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = scaler_y.inverse_transform(y_true)
    y_pred = scaler_y.inverse_transform(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def run_experiment(X, y, scaler_y, seq_length, hidden_dim=64, dropout_rate=0.0, 
                  use_mc_dropout=False, use_synaptic_pruning=False, batch_size=64, 
                  epochs=20, patience=5, num_trials=10, device='cuda'):
    
    results = {'mae': [], 'mse': [], 'rmse': [], 'train_loss': [], 'val_loss': [], 'epochs_run': [], 'runtime': []}
    X_seq, y_seq = create_sequences(X, y, seq_length)

    for trial in range(num_trials):
        trial_seed = 42 + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(trial_seed)
            torch.cuda.manual_seed_all(trial_seed)
        
        X_train, X_val, y_train, y_val = train_test_split_ts(X_seq, y_seq)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        input_dim = X_train.shape[2] if len(X_train.shape) == 3 else 1

        model = LSTM_Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout,
            use_synaptic_pruning=use_synaptic_pruning
        )
        
        model, train_losses, val_losses, best_val_loss, epochs_run, runtime = train_model(
            model, train_loader, val_loader, epochs=epochs, patience=patience, device=device
        )
        
        mae, mse, rmse = evaluate_model(model, val_loader, scaler_y, device)
        results['mae'].append(mae)
        results['mse'].append(mse)
        results['rmse'].append(rmse)
        results['train_loss'].append(train_losses[-1] if train_losses else float('inf'))
        results['val_loss'].append(best_val_loss)
        results['epochs_run'].append(epochs_run)
        results['runtime'].append(runtime)

    mae_array = np.array(results['mae'])
    mae_mean = np.mean(mae_array)
    mae_sem = stats.sem(mae_array) if len(mae_array) > 1 else 0
    confidence_interval = stats.t.interval(0.95, df=len(mae_array)-1, loc=mae_mean, scale=mae_sem) if len(mae_array) > 1 else (mae_mean, mae_mean)
    summary = {k: np.mean(v) for k, v in results.items()}
    summary['mae_95ci_low'] = confidence_interval[0]
    summary['mae_95ci_high'] = confidence_interval[1]
    return summary


def preprocess_data(df):
    # TODO: apply any pre-processing and feature-engineering here
    target_col = 'target'
    y = df[target_col].values
    df.drop(target_col, axis=1, inplace=True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler_X = StandardScaler()
    X_numeric_scaled = scaler_X.fit_transform(df[numeric_cols])
    df[numeric_cols] = X_numeric_scaled
    X = df.values
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    return X, y_scaled, scaler_X, scaler_y


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    df = pd.read_csv("SynapsePruning/Air_clean.csv")
    X, y, _, scaler_y = preprocess_data(df)
    
    epochs = 20
    trials = 10
    patience = 3
    sequence_lengths = [1, 3, 7, 14, 30, 60] 
    results = {}

    for seq_length in sequence_lengths:
        result_no_dropout = run_experiment(
            X, y, scaler_y, seq_length=seq_length, hidden_dim=64, 
            dropout_rate=0.0, use_mc_dropout=False, use_synaptic_pruning=False,
            epochs=epochs, patience=patience, num_trials=trials, device=device
        )

        result_dropout = run_experiment(
            X, y, scaler_y, seq_length=seq_length, hidden_dim=64, 
            dropout_rate=0.3, use_mc_dropout=False, use_synaptic_pruning=False,
            epochs=epochs, patience=patience, num_trials=trials, device=device
        )
        
        result_mc_dropout = run_experiment(
            X, y, scaler_y, seq_length=seq_length, hidden_dim=64, 
            dropout_rate=0.3, use_mc_dropout=True, use_synaptic_pruning=False,
            epochs=epochs, patience=patience, num_trials=trials, device=device
        )

        result_synaptic_pruning = run_experiment(
            X, y, scaler_y, seq_length=seq_length, hidden_dim=64, 
            dropout_rate=0.0, use_mc_dropout=False, use_synaptic_pruning=True,
            epochs=epochs, patience=patience, num_trials=trials, device=device
        )
        
        results[seq_length] = {
            'No Dropout': result_no_dropout,
            'Dropout': result_dropout,
            'MC Dropout': result_mc_dropout,
            'Synaptic Pruning': result_synaptic_pruning
        }
    
    print("\n\n" + "="*30 + " FINAL RESULTS " + "="*30)
    for seq_length, seq_results in results.items():
        print(f"\n=== Results for Sequence Length: {seq_length} ===")
        metrics = ['mae', 'rmse', 'val_loss', 'runtime', 'mae_95ci_low', 'mae_95ci_high']
        header = f"{'Metric':<15} {'No Dropout':<18} {'Dropout':<18} {'MC Dropout':<18} {'Synaptic Pruning':<18}"
        print(header)
        print('-' * len(header))
        for metric in metrics:
            no_drop_val = seq_results['No Dropout'][metric]
            drop_val = seq_results['Dropout'][metric]
            mc_drop_val = seq_results['MC Dropout'][metric]
            syn_prune_val = seq_results['Synaptic Pruning'][metric]
            print(f"{metric:<15} {no_drop_val:<18.4f} {drop_val:<18.4f} {mc_drop_val:<18.4f} {syn_prune_val:<18.4f}")


if __name__ == "__main__":
    main()