import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

#  读取预测结果
file_path = "ddg_predictions2.csv"
df = pd.read_csv(file_path)

#  确保包含 "EXP_DDG"（真实值）和 "DDG"（预测值）
if "EXP_DDG" not in df.columns or "DDG" not in df.columns:
    raise ValueError(" 文件缺少 'EXP_DDG' 或 'DDG' 列")

#  提取真实值和预测值
true_ddG = df["EXP_DDG"].values
pred_ddG = df["DDG"].values

#  计算 Spearman 相关系数
spearman_corr, _ = spearmanr(true_ddG, pred_ddG)

#  计算 Pearson 相关系数
pearson_corr, _ = pearsonr(true_ddG, pred_ddG)

#  计算 RMSE
rmse = np.sqrt(mean_squared_error(true_ddG, pred_ddG))

#  输出结果
print(f" Spearman 相关系数: {spearman_corr:.4f}")
print(f" Pearson 相关系数: {pearson_corr:.4f}")
print(f" RMSE: {rmse:.4f}")
