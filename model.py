import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    if 'ddG' not in data.columns:
        raise ValueError("CSV 文件中未找到 'ddG' 列，请确保标签列名为 'ddG'")
    X = data.drop(columns='ddG')
    y = data['ddG']
    return X, y

# 定义目标函数进行超参数优化
def objective(trial, X, y):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            param,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# 运行超参数优化
def optimize_hyperparameters(X, y):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    print(f"最佳超参数: {study.best_params}")
    return study.best_params

# 最终训练和保存模型
def train_and_save_model(X, y, best_params, output_path='lgbm_model.txt'):
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X, y)
    model.booster_.save_model(output_path)
    print(f"模型已保存至 {output_path}")

# 主函数
def main():
    file_path = 'merged_features.csv'
    X, y = load_data(file_path)
    best_params = optimize_hyperparameters(X, y)
    train_and_save_model(X, y, best_params)

if __name__ == '__main__':
    main()
