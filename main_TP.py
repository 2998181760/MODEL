import torch
from glob import glob
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb
import itertools

# ---------------------------- 配置 ----------------------------
model_save_path = 'esm2_650m_thermo_cycle_model.txt'
embedding_dir = 'esm2_650m_embeds'
thermo_output_dir = 'esm2_650m_embeds_thermo'
embedding_dict = {}

# 加载嵌入向量
for file in glob(f'{embedding_dir}/*.pt'):
    key = os.path.basename(file).split('.')[0]
    embedding_dict[key] = torch.load(file)['mean_representations'][33].detach().numpy()

# ---------------------------- 数据加载与预处理 ----------------------------
df_2648 = pd.read_csv('../S2648.tsv', sep='\t')
df_669 = pd.read_csv('../S669.tsv', sep='\t')

def parse_mutation(df):
    df['wt'] = df['mutation'].str[0]
    df['mut'] = df['mutation'].str[-1]
    df['position'] = df['mutation'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    return df

df_2648 = parse_mutation(df_2648)
df_669 = parse_mutation(df_669)

# ---------------------------- 热力学循环向量构建与保存 ----------------------------
def build_thermo_vectors(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    groups = df.groupby(["pdb", "chain", "position", "wt"])
    for (pdb, chain, pos, wtAA), group in groups:
        if len(group) < 2:
            continue
        for (_, row1), (_, row2) in itertools.permutations(group.iterrows(), 2):
            if row1['mut'] == row2['mut'] or row1['wt'] != row2['wt']:
                continue
            key1 = f"{pdb}_{chain}_{row1['mutation']}"
            key2 = f"{pdb}_{chain}_{row2['mutation']}"
            if key1 in embedding_dict and key2 in embedding_dict:
                vec = embedding_dict[key1] - embedding_dict[key2]
                label = row2['ddG'] - row1['ddG']
                torch.save({
                    'vector': vec,
                    'label': label
                }, os.path.join(output_dir, f"{key1}_to_{row2['mutation']}.pt"))

build_thermo_vectors(df_2648, thermo_output_dir)

# ---------------------------- 构建训练特征（原始 + 热力学） ----------------------------
X = []
y = []


# 热力学增强突变
for file in glob(f'{thermo_output_dir}/*.pt'):
    try:
        data = torch.load(file)
        X.append(data['vector'])
        y.append(data['label'])
    except:
        continue

X = np.array(X)
y = np.array(y)

# ---------------------------- 模型训练 ----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

lgb_params = {
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'max_depth': -1,
    'num_leaves': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'random_state': 42
}

model = lgb.LGBMRegressor(**lgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mse',
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# ---------------------------- 构建测试特征（原始 + 热力学） ----------------------------
X_test = []
y_test = []



# 热力学增强突变（测试集）
build_thermo_vectors(df_669, thermo_output_dir)
for file in glob(f'{thermo_output_dir}/*.pt'):
    try:
        data = torch.load(file)
        X_test.append(data['vector'])
        y_test.append(data['label'])
    except:
        continue

X_test = np.array(X_test)
y_test = np.array(y_test)

# ---------------------------- 模型评估 ----------------------------
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
