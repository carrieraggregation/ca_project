import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


class BiLSTMModel(nn.Module):
    def __init__(self, seq_len, num_seq_features, num_static_features,
                 hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2 + num_static_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_seq_features)
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        last_out = lstm_out[:, -1, :]
        combined = torch.cat([last_out, x_static], dim=1)
        return self.fc(combined)


def train_and_evaluate(args):
    # Determine data directory (python/data)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # python/models/bilstm
    data_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, 'data'))
    # Resolve feature and label paths
    X_path = args.X if os.path.isabs(args.X) else os.path.join(data_dir, args.X)
    y_path = args.y if os.path.isabs(args.y) else os.path.join(data_dir, args.y)

    # Load data using resolved paths
    data = np.load(X_path)
    labels = np.load(y_path)
    num_samples, total_dim = data.shape
    seq_part = args.history * args.numCC
    seq_len = args.history

    X_seq = data[:, :seq_part].reshape(num_samples, seq_len, args.numCC)
    X_stat = data[:, seq_part:]
    y = labels

    X_seq_t = torch.from_numpy(X_seq).float()
    X_stat_t = torch.from_numpy(X_stat).float()
    y_t = torch.from_numpy(y).float()

    # Dataset split
    dataset = TensorDataset(X_seq_t, X_stat_t, y_t)
    train_size = int(len(dataset) * (1 - args.val_split))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMModel(seq_len, args.numCC, X_stat.shape[1],
                        args.hidden_size, args.num_layers, args.dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train = 0.0
        for bx_seq, bx_stat, by in train_loader:
            bx_seq, bx_stat, by = bx_seq.to(device), bx_stat.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx_seq, bx_stat)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * bx_seq.size(0)
        train_loss = running_train / train_size
        train_losses.append(train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for vx_seq, vx_stat, vy in val_loader:
                vx_seq, vx_stat, vy = vx_seq.to(device), vx_stat.to(device), vy.to(device)
                vpreds = model(vx_seq, vx_stat)
                vloss = criterion(vpreds, vy)
                running_val += vloss.item() * vx_seq.size(0)
        val_loss = running_val / val_size
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = args.out_model if os.path.isabs(args.out_model) else os.path.join(script_dir, args.out_model)
            torch.save(model.state_dict(), model_path)

    # Ensure output directory
    # Plot loss curves
    loss_path = os.path.join(script_dir, 'loss_curve.png')
    plt.figure()
    plt.plot(range(1, args.epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training Curve')
    plt.savefig(loss_path)
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to {model_path}, loss curve saved as {loss_path}")

    # Validation metrics
    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for vx_seq, vx_stat, vy in val_loader:
            vx_seq, vx_stat = vx_seq.to(device), vx_stat.to(device)
            preds = model(vx_seq, vx_stat).cpu().numpy()
            all_pred.append(preds)
            all_true.append(vy.numpy())
    true_arr = np.vstack(all_true)
    pred_arr = np.vstack(all_pred)

    for cc in range(args.numCC):
        mae = mean_absolute_error(true_arr[:,cc], pred_arr[:,cc])
        r2  = r2_score(true_arr[:,cc], pred_arr[:,cc])
        print(f"CC{cc+1}: MAE = {mae:.3f}, R2 = {r2:.3f}")

    scatter_path = os.path.join(script_dir, 'cqi_scatter.png')
    plt.figure()
    plt.scatter(true_arr.flatten(), pred_arr.flatten(), alpha=0.3)
    mn, mx = true_arr.min(), true_arr.max()
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel('True CQI')
    plt.ylabel('Pred CQI')
    plt.title('True vs Pred CQI (Validation)')
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train & evaluate BI-LSTM for CQI prediction')
    parser.add_argument('--X', type=str, default='X.npy', help='Input features npy file')
    parser.add_argument('--y', type=str, default='y.npy', help='Target npy file')
    parser.add_argument('--history', type=int, default=10, help='History window size')
    parser.add_argument('--numCC', type=int, default=5, help='Number of CCs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='LSTM dropout')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--out_model', type=str, default='bilstm_cqi.pth', help='Output model file')
    args = parser.parse_args()

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_and_evaluate(args)
