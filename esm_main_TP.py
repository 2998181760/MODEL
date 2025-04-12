import torch
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb
import itertools

model_save_path = 'esm2_3b_mixed_model.txt'
embedding_dict = {}

# -------------------- 嵌入加载 --------------------
for file in glob('esm2_3B_embeds/*.pt'):
    key = file.split('/')[-1].split('.')[0]
    embedding_dict[key] = torch.load(file)['mean_representations'][36].detach().numpy()

# -------------------- 数据加载 --------------------
df_2648 = pd.read_csv('../S2648.tsv', sep='\t')
df_669 = pd.read_csv('../S669.tsv', sep='\t')

def parse_mutation(df):
    df['wt'] = df['mutation'].str[0]
    df['mut'] = df['mutation'].str[-1]
    df['position'] = df['mutation'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    return df

df_2648 = parse_mutation(df_2648)
df_669 = parse_mutation(df_669)

# -------------------- 原始特征（WT - MUT） --------------------
X_raw = []
y_raw = []

for _, row in df_2648.iterrows():
    wt_key = f'{row["pdb"]}_{row["chain"]}'
    mut_key = f'{row["pdb"]}_{row["chain"]}_{row["mutation"]}'
    if wt_key in embedding_dict and mut_key in embedding_dict:
        vec = embedding_dict[wt_key] - embedding_dict[mut_key]
        X_raw.append(vec)
        y_raw.append(row["ddG"])

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

# -------------------- 热力学特征（MUT1 - MUT2） --------------------
X_thermo = []
y_thermo = []

grouped = df_2648.groupby(['pdb', 'chain', 'position', 'wt'])
for (pdb, chain, pos, wt), group in grouped:
    if len(group) < 2:
        continue
    for (_, row1), (_, row2) in itertools.permutations(group.iterrows(), 2):
        if row1['mut'] == row2['mut']:
            continue
        key1 = f"{pdb}_{chain}_{row1['mutation']}"
        key2 = f"{pdb}_{chain}_{row2['mutation']}"
        if key1 in embedding_dict and key2 in embedding_dict:
            vec = embedding_dict[key1] - embedding_dict[key2]
            label = row2['ddG'] - row1['ddG']
            X_thermo.append(vec)
            y_thermo.append(label)

X_thermo = np.array(X_thermo)
y_thermo = np.array(y_thermo)

# -------------------- 特征拼接 + 划分训练集 --------------------
X_all = np.concatenate([X_raw, X_thermo], axis=0)
y_all = np.concatenate([y_raw, y_thermo], axis=0)

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=42)

# -------------------- 模型训练（添加超参数） --------------------
model = lgb.LGBMRegressor(
    learning_rate=0.01,
    n_estimators=1000,
    num_leaves=64,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mse'
)

# -------------------- 构建测试集特征（原始 + 热力学） --------------------
X_raw_test = []
y_raw_test = []

for _, row in df_669.iterrows():
    wt_key = f'{row["pdb"]}_{row["chain"]}'
    mut_key = f'{row["pdb"]}_{row["chain"]}_{row["mutation"]}'
    if wt_key in embedding_dict and mut_key in embedding_dict:
        vec = embedding_dict[wt_key] - embedding_dict[mut_key]
        X_raw_test.append(vec)
        y_raw_test.append(row["ddG"])

X_raw_test = np.array(X_raw_test)
y_raw_test = np.array(y_raw_test)

X_thermo_test = []
y_thermo_test = []

grouped_test = df_669.groupby(['pdb', 'chain', 'position', 'wt'])
for (pdb, chain, pos, wt), group in grouped_test:
    if len(group) < 2:
        continue
    for (_, row1), (_, row2) in itertools.permutations(group.iterrows(), 2):
        if row1['mut'] == row2['mut']:
            continue
        key1 = f"{pdb}_{chain}_{row1['mutation']}"
        key2 = f"{pdb}_{chain}_{row2['mutation']}"
        if key1 in embedding_dict and key2 in embedding_dict:
            vec = embedding_dict[key1] - embedding_dict[key2]
            label = row2['ddG'] - row1['ddG']
            X_thermo_test.append(vec)
            y_thermo_test.append(label)

X_thermo_test = np.array(X_thermo_test)
y_thermo_test = np.array(y_thermo_test)

# -------------------- 拼接测试集并评估 --------------------
X_test = np.concatenate([X_raw_test, X_thermo_test], axis=0)
y_test = np.concatenate([y_raw_test, y_thermo_test], axis=0)

model.booster_.save_model(model_save_path)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
pearson_corr, _ = pearsonr(y_test, preds)
spearman_corr, _ = spearmanr(y_test, preds)

print("\nTest Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"Pearson correlation: {pearson_corr:.4f}")
print(f"Spearman correlation: {spearman_corr:.4f}")
