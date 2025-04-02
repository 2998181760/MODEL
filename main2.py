import torch
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb

model_save_path = 'ESM2_650M_model_1.txt'
embedding_dict = {}

for file in glob('esm2_650m_embeds/*.pt'):
    embedding_dict[file.split('/')[-1].split('.')[0]] = torch.load(file)['mean_representations'][33].detach().numpy()

df_2648 = pd.read_csv('../S2648.tsv', sep='\t')
df_669 = pd.read_csv('../S669.tsv', sep='\t')

X = []
y = [] 
for i, row in df_2648.iterrows():
    try:
        wt = embedding_dict[f'{row["pdb"]}_{row["chain"]}']
        mut = embedding_dict[f'{row["pdb"]}_{row["chain"]}_{row["mutation"]}']
        X.append(wt - mut)
        y.append(row["ddG"])
    except KeyError:
        print(f'{row["pdb"]}_{row["chain"]}_{row["mutation"]}')

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=50)

# 添加超参数
params = {
    'learning_rate': 0.01,
    'n_estimators':1000,
    'num_leaves': 30,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'random_state': 50
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mse')

X_test = []
y_test = []
for i, row in df_669.iterrows():
    wt = embedding_dict[f'{row["pdb"]}_{row["chain"]}']
    mut = embedding_dict[f'{row["pdb"]}_{row["chain"]}_{row["mutation"]}']
    X_test.append(wt - mut)
    y_test.append(row["ddG"])

X_test = np.array(X_test)
y_test = np.array(y_test)

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
