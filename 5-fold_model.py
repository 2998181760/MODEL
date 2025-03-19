import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# 加载数据
data = pd.read_csv("merged_features.csv")
X = data.drop('ddG', axis=1)
y = data['ddG']

# 优化后的参数设置
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': -1,
    'min_data_in_leaf': 30,
    'lambda_l1': 0.2,
    'lambda_l2': 0.1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'seed': 42
}

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = []
best_iterations = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 使用train方法配合早停
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=0),
            lgb.log_evaluation(100)
        ]
    )
    
    best_iter = model.best_iteration
    best_iterations.append(best_iter)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    pearson = pearsonr(y_val, preds)[0]
    spearman = spearmanr(y_val, preds)[0]
    
    metrics.append((rmse, pearson, spearman))
    print(f"Fold {fold+1} | Best Iter: {best_iter} | RMSE: {rmse:.4f} | Pearson: {pearson:.4f} | Spearman: {spearman:.4f}")

# 输出平均指标
avg_metrics = np.mean(metrics, axis=0)
print(f"\nAverage Metrics | RMSE: {avg_metrics[0]:.4f} | Pearson: {avg_metrics[1]:.4f} | Spearman: {avg_metrics[2]:.4f}")

# 全数据训练最终模型（使用平均最佳迭代次数）
final_best_iter = int(np.mean(best_iterations))
print(f"\nTraining final model with {final_best_iter} iterations")

final_model = lgb.train(
    params,
    lgb.Dataset(X, label=y),
    num_boost_round=final_best_iter
)

# 保存模型
final_model.save_model('lgb_model_1.txt')

# 输出特征重要性
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print("\nTop 10 Features:\n", importance.head(10))