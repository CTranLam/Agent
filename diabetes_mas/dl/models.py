import torch.nn as nn


class DiabetesNet(nn.Module):
    """n_features -> 64 -> 32 -> 16 -> 8 -> 1. Trả về logit; xác suất = sigmoid(logit)."""

    def __init__(self, n_features: int, hidden: tuple = (64, 32, 16, 8), dropout: float = 0.1):
        super().__init__()
        dims = [n_features, *hidden]
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers += [nn.Linear(a, b), nn.ReLU()]
            if i < 2:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GlucoseLSTM(nn.Module):
    def __init__(self, hidden: int = 64, horizon: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, x):  # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # (B, horizon)
