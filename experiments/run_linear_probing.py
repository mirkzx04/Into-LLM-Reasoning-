import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pandas as pd
import wandb

import torch as th
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold

device = 'cuda' if th.cuda.is_available() else 'cpu'

layer = 0
probe_path = f"best_linear_probe_L{layer}.pt"
x_block = 'resid_pre'

DF_BASE_DIR = 'df_rlvr.pkl'


class LinearProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias = True)
    
    def forward(self, x):
        """
        X : Shape [B, d_model]
        """
        return self.proj(x)

def sclae_log(y):
    return th.sign(y) * th.log1p(th.abs(y))

def scale_origin(z) :
    return th.sign(z) * (th.exp(th.abs(z)) - 1)

def load_probe(d_model): 
    probe = LinearProbe(d_model).to(device)

    state_dict = th.load(
        probe_path,
        map_location=device,
        weights_only=True
    )
    probe.load_state_dict(state_dict)
    probe.eval()
    return probe

def compute_loss(
        z_true, 
        y_true, 
        z_pred, 
        y_pred, 
        lam = 0.7, 
        tau = 1e-8, 
        delta = 1.0
    ):
    abs_term = F.huber_loss(
        z_pred, z_true,
        reduction= "none", 
        delta=delta
    )

    rel_err = (y_pred - y_true) / (th.abs(y_true) + tau)
    rel_term = F.huber_loss(
        rel_err, th.zeros_like(rel_err), 
        reduction="none", 
        delta=delta
    )

    loss = lam * abs_term + (1 - lam) * rel_term
    return loss.mean()

def compute_acc_metrics(y_pred, y_true, eps = 1e-8, denom_offset = None):
    y_pred = y_pred.view(-1).float()
    y_true = y_true.view(-1).float()

    # Exact accuracy for integer targets
    exact_acc = (th.round(y_pred) == y_true).float().mean()

    abs_err = th.abs(y_pred - y_true)

    if denom_offset is None:
        denom = th.abs(y_true) + eps
    else:
        denom = th.abs(y_true) + denom_offset

    rel_err = abs_err / denom

    acc_1 = (rel_err <= 0.01).float().mean()
    acc_5 = (rel_err <= 0.05).float().mean()
    acc_10 = (rel_err <= 0.10).float().mean()

    medre = rel_err.median()
    mean_re = rel_err.mean()   

    return {
        "exact_acc": exact_acc.item(),
        "acc@1%": acc_1.item(),
        "acc@5%": acc_5.item(),
        "acc@10%": acc_10.item(),
        "medre": medre.item(),
        "mean_re": mean_re.item(),
    } 

def run_probe_inf(X, Y, probe, d_model, val = True):
    X, Y = X.to(device).float(), Y.to(device).float()

    if val:
        probe.eval() 
    else : 
        probe = load_probe(d_model, probe_path)

    with th.no_grad():

        z_true = sclae_log(Y)
        z_pred = probe(X)
        y_pred = scale_origin(z_pred)

    return compute_loss(z_true, Y, z_pred, y_pred).item(), compute_acc_metrics(y_pred, Y)

def train_probing(X_trn, Y_trn, X_val, Y_val, d_model, epochs = 100, lr = 1e-3, lam = 0.7):
    X_trn, X_val = X_trn.to(device).float(), X_val.to(device)
    Y_trn, Y_val = Y_trn.to(device).float(), Y_val.to(device)

    probe = LinearProbe(d_model).to(device)
    optimizer = opt.AdamW(probe.parameters(), lr = lr)

    best_loss = float('inf')

    for e in range(epochs): 
        probe.train()
        optimizer.zero_grad()

        z_true = sclae_log(Y_trn)
        z_pred = probe(X_trn)
        y_pred = scale_origin(z_pred)

        loss, train_metrics = compute_loss(z_true, Y_trn, z_pred, y_pred, lam=lam), compute_acc_metrics(y_pred, Y_trn)
        loss.backward()
        optimizer.step()

        val_loss, val_metrics = run_probe_inf(X_val, Y_val, d_model=d_model, val=True, probe=probe.eval())
        val_metrics['loss'] = val_loss

        if val_loss < best_loss: 
            best_loss = val_loss
            th.save(probe.state_dict())

    return val_metrics

def build_X_Y(df):
    X_list = []
    Y_list = []

    for i in range(len(df)):
        x = df.loc[i, x_block][layer].view(-1) # Shape : [d_model]
        y = float(str(df.loc[i, 'ground_truth']).replace(',', '').strip())    

        X_list.append(x)
        Y_list.append(y)

    X = th.stack(X_list) # Shape : [B, d_model]
    Y = th.tensor(Y_list).unsqueeze(1) # Shape : [B, 1]    

    return X, Y

def k_fold_cv(X, Y, d_model, n_splits = 5, epochs = 100, lr=1e-3, lam=0.7):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    for _, (trainx_idx, val_idx) in enumerate(kf.split(X), start= 1):
        X_trn, X_val = X[trainx_idx], X[val_idx]
        Y_trn, Y_val = Y[trainx_idx], Y[val_idx]

        metrics = train_probing(
            X_trn, Y_trn, X_val, Y_val, 
            d_model=d_model, 
            epochs=epochs,
            lr = lr, 
            lam = lam
        )

        fold_metrics.append(metrics)

    mean_metrics = {}
    for k in fold_metrics[0].keys():
        mean_metrics[k] = sum(m[k] for m in fold_metrics) / len(fold_metrics)

    return mean_metrics

df_base = pd.read_pickle(DF_BASE_DIR)
X, Y = build_X_Y(df_base)
