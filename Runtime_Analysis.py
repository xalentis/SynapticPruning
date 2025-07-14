# Gideon Vos 2025
# James Cook University, Australia
# www.linkedin/in/gideonvos

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import random
import scipy.stats as stats
import auto_feature_engineering
import matplotlib.pyplot as plt
import psutil
import os


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
                max_prune_rate=0.7,
                min_prune_rate=0.3,
                warmup_epochs=2,
                prune_every=5,
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
        if target_sparsity <= 0: return

        all_weights, param_info = [], []
        for module in self.modules:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    active_indices = self.masks[param].bool()
                    active_weights = param.data[active_indices]
                    if active_weights.numel() > 0:
                        all_weights.append(torch.abs(active_weights))
                        param_info.append((param, active_indices))

        if not all_weights: return
            
        all_weights_tensor = torch.cat(all_weights)
        total_weights = sum(p.numel() for p, _ in param_info)
        current_active_weights = all_weights_tensor.numel()
        target_total_pruned = int(target_sparsity * total_weights)
        currently_pruned = total_weights - current_active_weights
        additional_to_prune = max(0, target_total_pruned - currently_pruned)
        
        if additional_to_prune == 0 or additional_to_prune >= current_active_weights: return

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
        should_prune = (self.epoch >= self.warmup_epochs and self.batches_processed % self.prune_every == 0)
        if should_prune:
            self._update_masks()
            self._apply_pruning_masks()
            return True
        return False
        
    def get_sparsity_stats(self):
        stats_dict = {}
        for module in self.modules:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    mask = self.masks[param]
                    sparsity = (mask == 0).sum().item() / mask.numel()
                    stats_dict[f"{module.__class__.__name__}_{name}"] = {'sparsity': sparsity}
        return stats_dict


def setup_plot_style():
    font = {'family': 'Times New Roman', 'weight': 'bold'}
    plt.rc('font', **font)
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'


def plot_results(results):
    setup_plot_style()
    methods = list(next(iter(results.values())).keys())
    seq_lengths = list(results.keys())
    metrics_to_plot = {
        'mae': 'Mean Absolute Error (MAE)',
        'rmse': 'Root Mean Squared Error (RMSE)',
        'runtime': 'Average Runtime (seconds)',
        'peak_memory_mb': 'Peak Memory Usage (MB)'
    }
    for metric, title in metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        for method in methods:
            metric_values = [results[seq][method][metric] for seq in seq_lengths]
            plt.plot(seq_lengths, metric_values, marker='o', linestyle='-', label=method)
        plt.xlabel("Sequence Length")
        plt.ylabel(title)
        plt.title(f"{title} vs. Sequence Length", fontsize=16, fontweight='bold')
        plt.legend()
        plt.xticks(seq_lengths)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plot_filename = f"{metric}_comparison_summary.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()


def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    if len(X.shape) == 1: X = X.reshape(-1, 1)
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)


def train_test_split_ts(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def early_stopping(val_losses, patience=5):
    if len(val_losses) < patience + 1: return False
    for i in range(patience):
        if val_losses[-(i+1)] < val_losses[-(i+2)]: return False
    return True


def train_model(model, train_loader, val_loader, epochs=20, patience=3, device='cuda'):
    model.to(device)
    if model.use_synaptic_pruning and model.synaptic_pruning:
        model.synaptic_pruning.set_device(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
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
        train_losses.append(train_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                if model.use_mc_dropout: model.train()
                val_outputs = model(X_val, y_val, epoch)
                val_loss += criterion(val_outputs, y_val.unsqueeze(1)).item() * X_val.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss: best_val_loss = val_loss
        if early_stopping(val_losses, patience):
            break

    training_time = time.time() - start_time
    peak_memory_mb = 0
    if device == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    else:
        process = psutil.Process(os.getpid())
        peak_memory_bytes = process.memory_info().rss
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    return model, train_losses, val_losses, best_val_loss, epoch + 1, training_time, peak_memory_mb


def evaluate_model(model, test_loader, scaler_y, device='cuda'):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            if model.use_mc_dropout: model.train()
            outputs = model(X_test)
            y_true.extend(y_test.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy().flatten())
    y_true = scaler_y.inverse_transform(np.array(y_true).reshape(-1, 1))
    y_pred = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1))
    return mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))


def run_experiment(X, y, scaler_y, seq_length, hidden_dim=64, dropout_rate=0.0, 
                  use_mc_dropout=False, use_synaptic_pruning=False, batch_size=64, 
                  epochs=20, patience=5, num_trials=10, device='cuda'):
    
    results = {'mae': [], 'mse': [], 'rmse': [], 'val_loss': [], 'runtime': [], 'peak_memory_mb': []}
    X_seq, y_seq = create_sequences(X, y, seq_length)

    for trial in range(num_trials):
        trial_seed = 42 + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)
        
        X_train, X_val, y_train, y_val = train_test_split_ts(X_seq, y_seq)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = LSTM_Forecaster(
            input_dim=X_train.shape[2], hidden_dim=hidden_dim, dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout, use_synaptic_pruning=use_synaptic_pruning
        )
        
        model, _, _, best_val_loss, _, runtime, peak_memory_mb = train_model(
            model, train_loader, val_loader, epochs=epochs, patience=patience, device=device
        )
        
        mae, mse, rmse = evaluate_model(model, val_loader, scaler_y, device)
        results['mae'].append(mae); results['mse'].append(mse); results['rmse'].append(rmse)
        results['val_loss'].append(best_val_loss); results['runtime'].append(runtime); results['peak_memory_mb'].append(peak_memory_mb)

    summary = {k: np.mean(v) for k, v in results.items()}
    mae_array = np.array(results['mae'])
    if len(mae_array) > 1:
        mae_mean = np.mean(mae_array)
        mae_sem = stats.sem(mae_array)
        ci = stats.t.interval(0.95, df=len(mae_array)-1, loc=mae_mean, scale=mae_sem)
        summary['mae_95ci_low'], summary['mae_95ci_high'] = ci
    else:
        summary['mae_95ci_low'], summary['mae_95ci_high'] = (summary['mae'], summary['mae'])
        
    return summary


def preprocess_data(df):
    df_processed = df.copy()
    (df_processed, _) = auto_feature_engineering.auto_feature_engineering(
        df_processed, date_columns=["timeOpen"], sequential_date_columns=["timeOpen"],
        lag_periods=[1,3,7,14,30], lead_periods=[1,3,7,14,30], target_column="close"
    )
    y = df_processed['close'].values
    df_processed.drop('close', axis=1, inplace=True)
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    scaler_X = StandardScaler()
    df_processed[numeric_cols] = scaler_X.fit_transform(df_processed[numeric_cols])
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    return df_processed.values, y_scaled, scaler_X, scaler_y


def main():
    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    df = pd.read_csv("bitcoin.csv")
    X, y, _, scaler_y = preprocess_data(df)
    
    epochs, trials, patience = 20, 10, 3
    sequence_lengths = [1, 3, 7, 14, 30, 60] 
    results = {}

    for seq_length in sequence_lengths:
        print(f"\n{'='*25}\nRunning experiments with sequence length: {seq_length}\n{'='*25}")
        results[seq_length] = {
            'No Dropout': run_experiment(X, y, scaler_y, seq_length, dropout_rate=0.0, use_mc_dropout=False, use_synaptic_pruning=False, epochs=epochs, patience=patience, num_trials=trials, device=device),
            'Dropout': run_experiment(X, y, scaler_y, seq_length, dropout_rate=0.3, use_mc_dropout=False, use_synaptic_pruning=False, epochs=epochs, patience=patience, num_trials=trials, device=device),
            'MC Dropout': run_experiment(X, y, scaler_y, seq_length, dropout_rate=0.3, use_mc_dropout=True, use_synaptic_pruning=False, epochs=epochs, patience=patience, num_trials=trials, device=device),
            'Synaptic Pruning': run_experiment(X, y, scaler_y, seq_length, dropout_rate=0.0, use_mc_dropout=False, use_synaptic_pruning=True, epochs=epochs, patience=patience, num_trials=trials, device=device)
        }
    
    print("\n\n" + "="*35 + " FINAL STATISTICAL SUMMARY " + "="*35)
    for seq_length, seq_results in results.items():
        print(f"\n=== Results for Sequence Length: {seq_length} ===")
        metrics = ['mae', 'rmse', 'runtime', 'peak_memory_mb', 'mae_95ci_low', 'mae_95ci_high']
        header = f"{'Metric':<18} {'No Dropout':<18} {'Dropout':<18} {'MC Dropout':<18} {'Synaptic Pruning':<18}"
        print(header); print('-' * len(header))
        for metric in metrics:
            if all(metric in res for res in seq_results.values()):
                vals = [seq_results[method][metric] for method in seq_results]
                print(f"{metric:<18} {vals[0]:<18.4f} {vals[1]:<18.4f} {vals[2]:<18.4f} {vals[3]:<18.4f}")

    print("\n\n" + "="*35 + " GENERATING PLOTS " + "="*35)
    plot_results(results)

if __name__ == "__main__":
    main()