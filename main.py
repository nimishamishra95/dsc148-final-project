"""
DSC 148 Final Project
NBA Game Outcome Prediction via Rolling Team Statistics
Author: Nimisha Mishra
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("figures", exist_ok=True)

PANEL_BG = "#161b22"; ACC = "#58a6ff"; ACC2 = "#f78166"; TEXT_COL = "#e6edf3"

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL); ax.yaxis.label.set_color(TEXT_COL)
    if title: ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=8)


# LOAD
print("=" * 60)
print("STEP 1 - Loading data")
print("=" * 60)

games   = pd.read_csv("data/games.csv")
details = pd.read_csv("data/games_details.csv")

games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])
games = games.sort_values("GAME_DATE_EST").reset_index(drop=True)
games = games.dropna(subset=["HOME_TEAM_WINS"])
games["HOME_TEAM_WINS"] = games["HOME_TEAM_WINS"].astype(int)

print(f"  games.csv       : {games.shape[0]:,} rows")
print(f"  games_details   : {details.shape[0]:,} rows")
print(f"  Date range      : {games['GAME_DATE_EST'].min().date()} - {games['GAME_DATE_EST'].max().date()}")
print(f"  Home win %%      : {games['HOME_TEAM_WINS'].mean()*100:.1f}%%")


# BUILD REAL TEAM-GAME STATS FROM games_details.csv

print("\n" + "=" * 60)
print("STEP 2 - Aggregating player box scores into team stats")
print("=" * 60)

# Parse minutes (MM:SS or float)
def parse_min(m):
    try:
        if pd.isna(m): return 0.0
        s = str(m)
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(s)
    except: return 0.0

details["MIN_NUM"] = details["MIN"].apply(parse_min)

# Keep only players who actually played
det = details[details["MIN_NUM"] > 0].copy()

# Fill numeric NaNs with 0
NUM_COLS = ["PTS","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","REB",
            "AST","STL","BLK","TO","PF","PLUS_MINUS"]
for c in NUM_COLS:
    if c in det.columns:
        det[c] = pd.to_numeric(det[c], errors="coerce").fillna(0)

# Aggregate to team level
team_game = det.groupby(["GAME_ID", "TEAM_ID"]).agg(
    PTS       = ("PTS",       "sum"),
    FGM       = ("FGM",       "sum"),
    FGA       = ("FGA",       "sum"),
    FG3M      = ("FG3M",      "sum"),
    FG3A      = ("FG3A",      "sum"),
    FTM       = ("FTM",       "sum"),
    FTA       = ("FTA",       "sum"),
    OREB      = ("OREB",      "sum"),
    DREB      = ("DREB",      "sum"),
    REB       = ("REB",       "sum"),
    AST       = ("AST",       "sum"),
    STL       = ("STL",       "sum"),
    BLK       = ("BLK",       "sum"),
    TO        = ("TO",        "sum"),
    PF        = ("PF",        "sum"),
    PLUS_MINUS= ("PLUS_MINUS","sum"),
).reset_index()

# Compute efficiency stats
team_game["FG_PCT"]  = team_game["FGM"]  / (team_game["FGA"]  + 1e-5)
team_game["FG3_PCT"] = team_game["FG3M"] / (team_game["FG3A"] + 1e-5)
team_game["FT_PCT"]  = team_game["FTM"]  / (team_game["FTA"]  + 1e-5)
team_game["AST_TO"]  = team_game["AST"]  / (team_game["TO"]   + 1e-5)
team_game["TRUE_SHOOTING"] = team_game["PTS"] / (
    2 * (team_game["FGA"] + 0.44 * team_game["FTA"]) + 1e-5)

STAT_COLS = ["PTS","FG_PCT","FG3_PCT","FT_PCT","REB","OREB","DREB",
             "AST","STL","BLK","TO","AST_TO","TRUE_SHOOTING","PLUS_MINUS"]

print(f"  Team-game rows  : {len(team_game):,}")
print(f"  Stats per team  : {len(STAT_COLS)}")

# Join date from games
game_dates = games[["GAME_ID","GAME_DATE_EST","HOME_TEAM_ID","VISITOR_TEAM_ID","HOME_TEAM_WINS"]]
team_game  = team_game.merge(game_dates, on="GAME_ID", how="inner")
team_game  = team_game.sort_values(["TEAM_ID","GAME_DATE_EST"]).reset_index(drop=True)


# EDA
print("\n" + "=" * 60)
print("STEP 3 - Exploratory Data Analysis")
print("=" * 60)

# Tag win/loss for each team-game row
team_game["WIN"] = (
    ((team_game["TEAM_ID"] == team_game["HOME_TEAM_ID"])    & (team_game["HOME_TEAM_WINS"] == 1)) |
    ((team_game["TEAM_ID"] == team_game["VISITOR_TEAM_ID"]) & (team_game["HOME_TEAM_WINS"] == 0))
).astype(int)

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Home win rate by season
season_hw = games.groupby("SEASON")["HOME_TEAM_WINS"].mean().reset_index()
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(season_hw["SEASON"].astype(str), season_hw["HOME_TEAM_WINS"],
         marker="o", color=ACC, linewidth=2, markersize=5)
ax1.axhline(0.5, color=ACC2, linestyle="--", linewidth=1, alpha=0.7)
ax1.set_ylim(0.4, 0.7); ax1.set_xlabel("Season"); ax1.set_ylabel("Home Win Rate")
ax1.tick_params(axis="x", rotation=45, labelsize=7)
style_ax(ax1, "Home Win Rate by Season")

# Panel 2: PTS distribution win vs loss
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(team_game[team_game["WIN"]==1]["PTS"], bins=40, alpha=0.7, color=ACC,  label="Win",  edgecolor="none")
ax2.hist(team_game[team_game["WIN"]==0]["PTS"], bins=40, alpha=0.7, color=ACC2, label="Loss", edgecolor="none")
ax2.set_xlabel("Points Scored"); ax2.set_ylabel("Frequency")
ax2.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COL, fontsize=8)
style_ax(ax2, "Points: Win vs Loss")

# Panel 3: True Shooting % by outcome
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(team_game[team_game["WIN"]==1]["TRUE_SHOOTING"].clip(0,1), bins=40, alpha=0.7, color=ACC,  label="Win",  edgecolor="none")
ax3.hist(team_game[team_game["WIN"]==0]["TRUE_SHOOTING"].clip(0,1), bins=40, alpha=0.7, color=ACC2, label="Loss", edgecolor="none")
ax3.set_xlabel("True Shooting %"); ax3.set_ylabel("Frequency")
ax3.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COL, fontsize=8)
style_ax(ax3, "True Shooting %%: Win vs Loss")

# Panel 4: AST/TO ratio
ax4 = fig.add_subplot(gs[1, 0])
w = team_game[team_game["WIN"]==1]["AST_TO"].clip(0, 8)
l = team_game[team_game["WIN"]==0]["AST_TO"].clip(0, 8)
ax4.boxplot([w, l], patch_artist=True,
            boxprops=dict(facecolor=PANEL_BG, color=ACC),
            medianprops=dict(color=ACC2, linewidth=2),
            whiskerprops=dict(color=TEXT_COL), capprops=dict(color=TEXT_COL),
            flierprops=dict(marker=".", color=ACC, alpha=0.1, markersize=2))
ax4.set_xticklabels(["Win", "Loss"]); ax4.set_ylabel("AST/TO Ratio")
style_ax(ax4, "Assist/Turnover Ratio by Outcome")

# Panel 5: STL + BLK (defensive stats)
team_game["DEF"] = team_game["STL"] + team_game["BLK"]
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(team_game[team_game["WIN"]==1]["DEF"], bins=30, alpha=0.7, color=ACC,  label="Win",  edgecolor="none")
ax5.hist(team_game[team_game["WIN"]==0]["DEF"], bins=30, alpha=0.7, color=ACC2, label="Loss", edgecolor="none")
ax5.set_xlabel("STL + BLK"); ax5.set_ylabel("Frequency")
ax5.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COL, fontsize=8)
style_ax(ax5, "Defensive Stats (STL+BLK) by Outcome")

# Panel 6: Correlation heatmap of key stats
ax6 = fig.add_subplot(gs[1, 2])
corr_cols = ["PTS","TRUE_SHOOTING","AST_TO","REB","STL","BLK","TO"]
corr = team_game[corr_cols].corr()
sns.heatmap(corr, ax=ax6, cmap="coolwarm", center=0, annot=True, fmt=".1f",
            annot_kws={"size": 6, "color": "white"}, linecolor="#30363d", linewidths=0.3,
            cbar_kws={"shrink": 0.7})
ax6.tick_params(colors=TEXT_COL, labelsize=7)
style_ax(ax6, "Feature Correlation")

fig.suptitle("NBA Game Data - EDA from Player Box Scores (2004-2022)",
             color=TEXT_COL, fontsize=13, fontweight="bold", y=1.01)
plt.savefig("figures/eda.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved to figures/eda.png")


# ROLLING FEATURE ENGINEERING
print("\n" + "=" * 60)
print("STEP 4 - Rolling Feature Engineering")
print("=" * 60)

WINDOW = 10

# For each team, compute rolling mean of last WINDOW games
roll_frames = []
for col in STAT_COLS:
    roll_frames.append(
        team_game.groupby("TEAM_ID")[col]
                 .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=4).mean())
                 .rename(f"ROLL_{col}")
    )
team_game = pd.concat([team_game] + roll_frames, axis=1)

roll_cols = [f"ROLL_{c}" for c in STAT_COLS]

# Split into home / away perspectives
home_tg = team_game[team_game["TEAM_ID"] == team_game["HOME_TEAM_ID"]].copy()
away_tg = team_game[team_game["TEAM_ID"] == team_game["VISITOR_TEAM_ID"]].copy()

home_feats = home_tg.set_index("GAME_ID")[roll_cols].add_prefix("H_")
away_feats = away_tg.set_index("GAME_ID")[roll_cols].add_prefix("A_")

merged = home_feats.join(away_feats, how="inner")

# Difference features
for col in roll_cols:
    merged[f"DIFF_{col}"] = merged[f"H_{col}"] - merged[f"A_{col}"]

# Home court flag
merged["HOME_COURT"] = 1.0

labels = games.set_index("GAME_ID")["HOME_TEAM_WINS"]
merged = merged.join(labels, how="inner").dropna()

feature_cols = [c for c in merged.columns if c != "HOME_TEAM_WINS"]
X = merged[feature_cols].values.astype(np.float32)
y = merged["HOME_TEAM_WINS"].values.astype(np.int64)

print(f"  Samples         : {X.shape[0]:,}")
print(f"  Features        : {X.shape[1]}  ({len(STAT_COLS)} home + {len(STAT_COLS)} away + {len(STAT_COLS)} diff + 1 court)")
print(f"  Class balance   : {y.mean()*100:.1f}%% home wins")


# TEMPORAL TRAIN/VAL/TEST SPLIT

n = len(X)
tr_end = int(n * 0.70); va_end = int(n * 0.85)
X_train, y_train = X[:tr_end],      y[:tr_end]
X_val,   y_val   = X[tr_end:va_end],y[tr_end:va_end]
X_test,  y_test  = X[va_end:],      y[va_end:]

scaler = StandardScaler().fit(X_train)
Xtr_s  = scaler.transform(X_train)
Xva_s  = scaler.transform(X_val)
Xte_s  = scaler.transform(X_test)

print(f"  Train : {len(X_train):,}  |  Val : {len(X_val):,}  |  Test : {len(X_test):,}")


# BASELINES

print("\n" + "=" * 60)
print("STEP 5 - Baselines")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
lr.fit(Xtr_s, y_train)
lr_pred  = lr.predict(Xte_s)
lr_proba = lr.predict_proba(Xte_s)[:, 1]

nb = GaussianNB()
nb.fit(Xtr_s, y_train)
nb_pred  = nb.predict(Xte_s)
nb_proba = nb.predict_proba(Xte_s)[:, 1]

for name, pred, proba in [("Logistic Regression", lr_pred, lr_proba),
                            ("Naive Bayes",        nb_pred, nb_proba)]:
    print(f"  {name:<22}: Acc={accuracy_score(y_test,pred):.4f}  "
          f"F1={f1_score(y_test,pred):.4f}  AUC={roc_auc_score(y_test,proba):.4f}")


# MLP

print("\n" + "=" * 60)
print("STEP 6 - Training MLP")
print("=" * 60)

class NBAMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []; prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

def make_loader(X_np, y_np, batch=256, shuffle=False):
    ds = TensorDataset(torch.tensor(X_np), torch.tensor(y_np, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = NBAMLP(X.shape[1]).to(device)
opt       = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
crit      = nn.BCEWithLogitsLoss()

EPOCHS = 60
train_losses, val_losses = [], []
best_val, best_state = float("inf"), None

tr_loader = make_loader(Xtr_s, y_train, shuffle=True)
va_loader = make_loader(Xva_s, y_val)
te_loader = make_loader(Xte_s, y_test)

for epoch in range(1, EPOCHS + 1):
    model.train(); tl = 0
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        tl += loss.item() * len(xb)
    tl /= len(Xtr_s)

    model.eval(); vl = 0
    with torch.no_grad():
        for xb, yb in va_loader:
            vl += crit(model(xb.to(device)), yb.to(device)).item() * len(xb)
    vl /= len(Xva_s)
    train_losses.append(tl); val_losses.append(vl); sched.step()
    if vl < best_val: best_val = vl; best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  train={tl:.4f}  val={vl:.4f}")

model.load_state_dict(best_state); model.eval()
mlp_logits, mlp_true = [], []
with torch.no_grad():
    for xb, yb in te_loader:
        mlp_logits.append(model(xb.to(device)).cpu()); mlp_true.append(yb)
mlp_logits = torch.cat(mlp_logits).numpy()
mlp_proba  = torch.sigmoid(torch.tensor(mlp_logits)).numpy()
mlp_pred   = (mlp_proba >= 0.5).astype(int)
mlp_true   = torch.cat(mlp_true).numpy().astype(int)

mlp_acc = accuracy_score(mlp_true, mlp_pred)
mlp_f1  = f1_score(mlp_true, mlp_pred)
mlp_auc = roc_auc_score(mlp_true, mlp_proba)
print(f"\n  MLP (proposed) : Acc={mlp_acc:.4f}  F1={mlp_f1:.4f}  AUC={mlp_auc:.4f}")


# RESULTS FIGURES

print("\n" + "=" * 60)
print("STEP 7 - Result figures")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0d1117")

# Training curve
ax = axes[0]; ax.set_facecolor(PANEL_BG)
ax.plot(train_losses, color=ACC, label="Train"); ax.plot(val_losses, color=ACC2, label="Val")
ax.set_xlabel("Epoch", color=TEXT_COL); ax.set_ylabel("BCE Loss", color=TEXT_COL)
ax.set_title("MLP Training Curve", color=TEXT_COL, fontweight="bold")
ax.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COL)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
ax.tick_params(colors=TEXT_COL)

# Model comparison bar chart
ax = axes[1]; ax.set_facecolor(PANEL_BG)
mnames = ["Naive\nBayes", "Logistic\nRegression", "MLP\n(Ours)"]
accs_  = [accuracy_score(y_test,nb_pred), accuracy_score(y_test,lr_pred), mlp_acc]
f1s_   = [f1_score(y_test,nb_pred),       f1_score(y_test,lr_pred),       mlp_f1]
aucs_  = [roc_auc_score(y_test,nb_proba), roc_auc_score(y_test,lr_proba), mlp_auc]
x_ = np.arange(3); w_ = 0.25
b1 = ax.bar(x_-w_, accs_, w_, label="Accuracy",  color=ACC,       edgecolor="none")
b2 = ax.bar(x_,    f1s_,  w_, label="F1",        color="#3fb950", edgecolor="none")
b3 = ax.bar(x_+w_, aucs_, w_, label="AUC-ROC",   color=ACC2,      edgecolor="none")
ax.set_xticks(x_); ax.set_xticklabels(mnames, color=TEXT_COL, fontsize=9)
ax.set_ylim(0.5, 0.85); ax.set_ylabel("Score", color=TEXT_COL)
ax.set_title("Model Comparison", color=TEXT_COL, fontweight="bold")
ax.legend(facecolor=PANEL_BG, edgecolor="#30363d", labelcolor=TEXT_COL, fontsize=8)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
ax.tick_params(colors=TEXT_COL)
for bars in [b1,b2,b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.003, f"{h:.3f}",
                ha="center", va="bottom", color=TEXT_COL, fontsize=7)

# Confusion matrix
ax = axes[2]; ax.set_facecolor(PANEL_BG)
cm = confusion_matrix(mlp_true, mlp_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            annot_kws={"color":"white","size":14}, cbar_kws={"shrink":0.8},
            linecolor="#30363d", linewidths=0.5)
ax.set_xlabel("Predicted", color=TEXT_COL); ax.set_ylabel("Actual", color=TEXT_COL)
ax.set_xticklabels(["Away Win","Home Win"], color=TEXT_COL)
ax.set_yticklabels(["Away Win","Home Win"], color=TEXT_COL, rotation=0)
ax.set_title("MLP Confusion Matrix", color=TEXT_COL, fontweight="bold")

fig.suptitle("NBA Outcome Prediction - Results (Box Score Features)", color=TEXT_COL,
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved to figures/results.png")


# ABLATION STUDY

print("\n" + "=" * 60)
print("STEP 8 - Ablation Study")
print("=" * 60)

n_s = len(STAT_COLS)
idx_home = list(range(n_s))
idx_away = list(range(n_s, 2*n_s))
idx_diff = list(range(2*n_s, 3*n_s))
idx_all  = list(range(3*n_s + 1))

def eval_subset(idx, name, epochs=35):
    Xtr = Xtr_s[:, idx]; Xva = Xva_s[:, idx]; Xte = Xte_s[:, idx]
    m   = NBAMLP(len(idx)).to(device)
    o   = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best_v, best_s = float("inf"), None
    for _ in range(epochs):
        m.train()
        for xb, yb in make_loader(Xtr, y_train, shuffle=True):
            o.zero_grad(); crit(m(xb.to(device)), yb.to(device)).backward(); o.step()
        m.eval(); vl = 0
        with torch.no_grad():
            for xb, yb in make_loader(Xva, y_val):
                vl += crit(m(xb.to(device)), yb.to(device)).item() * len(xb)
        vl /= len(Xva)
        if vl < best_v: best_v = vl; best_s = {k: v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_s); m.eval()
    lgs = []
    with torch.no_grad():
        for xb, _ in make_loader(Xte, y_test): lgs.append(m(xb.to(device)).cpu())
    pr = torch.sigmoid(torch.cat(lgs)).numpy(); pd_ = (pr>=0.5).astype(int)
    acc = accuracy_score(y_test, pd_); auc = roc_auc_score(y_test, pr)
    print(f"    {name:<38}: Acc={acc:.4f}  AUC={auc:.4f}")
    return acc, auc

eval_subset(idx_home,           "Home rolling only")
eval_subset(idx_away,           "Away rolling only")
eval_subset(idx_diff,           "Difference only")
eval_subset(idx_home+idx_away,  "Home + Away (no diff)")
eval_subset(idx_all,            "All features (full model)")


# WINDOW SENSITIVITY

print("\n" + "=" * 60)
print("STEP 9 - Window sensitivity")
print("=" * 60)

window_res = []
for w in [3, 5, 10, 15, 20]:
    rf = []
    for col in STAT_COLS:
        rf.append(
            team_game.groupby("TEAM_ID")[col]
                     .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=3).mean())
                     .rename(f"ROLL_{col}")
        )
    tg2 = pd.concat([team_game[["GAME_ID","TEAM_ID","HOME_TEAM_ID","VISITOR_TEAM_ID"]]] + rf, axis=1)
    h2 = tg2[tg2["TEAM_ID"]==tg2["HOME_TEAM_ID"]].set_index("GAME_ID")[[f"ROLL_{c}" for c in STAT_COLS]].add_prefix("H_")
    a2 = tg2[tg2["TEAM_ID"]==tg2["VISITOR_TEAM_ID"]].set_index("GAME_ID")[[f"ROLL_{c}" for c in STAT_COLS]].add_prefix("A_")
    m2 = h2.join(a2, how="inner")
    for col in [f"ROLL_{c}" for c in STAT_COLS]:
        m2[f"D_{col}"] = m2[f"H_{col}"] - m2[f"A_{col}"]
    m2 = m2.join(labels, how="inner").dropna()
    fc = [c for c in m2.columns if c != "HOME_TEAM_WINS"]
    X2 = m2[fc].values.astype(np.float32); y2 = m2["HOME_TEAM_WINS"].values.astype(np.int64)
    n2 = len(X2); t2 = int(n2*0.7); v2 = int(n2*0.85)
    sc2 = StandardScaler().fit(X2[:t2])
    lrw = LogisticRegression(max_iter=500, random_state=SEED).fit(sc2.transform(X2[:t2]), y2[:t2])
    acc = accuracy_score(y2[v2:], lrw.predict(sc2.transform(X2[v2:])))
    window_res.append((w, acc))
    print(f"    Window={w:2d}  -  Val Acc={acc:.4f}")

fig, ax = plt.subplots(figsize=(6,4))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor(PANEL_BG)
ws, accs_w = zip(*window_res)
ax.plot(ws, accs_w, marker="o", color=ACC, linewidth=2, markersize=7)
ax.set_xlabel("Rolling Window Size", color=TEXT_COL); ax.set_ylabel("Val Accuracy", color=TEXT_COL)
ax.set_title("Sensitivity to Window Size", color=TEXT_COL, fontweight="bold")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
ax.tick_params(colors=TEXT_COL)
plt.tight_layout()
plt.savefig("figures/window_sensitivity.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved to figures/window_sensitivity.png")


# SUMMARY
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
rows = [
    ("Naive Bayes",         accuracy_score(y_test,nb_pred), f1_score(y_test,nb_pred), roc_auc_score(y_test,nb_proba)),
    ("Logistic Regression", accuracy_score(y_test,lr_pred), f1_score(y_test,lr_pred), roc_auc_score(y_test,lr_proba)),
    ("MLP (proposed)",      mlp_acc, mlp_f1, mlp_auc),
]
print(f"  {'Model':<24} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10}")
print("  " + "-"*56)
for name, acc, f1, auc in rows:
    print(f"  {name:<24} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")
print("=" * 60)
print("\nDone! Figures saved to ./figures/")