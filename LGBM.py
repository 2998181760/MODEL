import os
import torch
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

# --------------------------
# 增强型特征处理器（修复版）
# --------------------------
class FeatureEngineer:
    def __init__(self, max_components=50):
        self.scaler = StandardScaler()
        self.pca = PCA(svd_solver='auto')
        self.max_components = max_components
        self.is_fitted = False

    def _safe_extract_tensors(self, obj):
        """安全递归提取张量"""
        try:
            if isinstance(obj, torch.Tensor):
                return [obj.numpy().flatten()]
            if isinstance(obj, dict):
                return [v for value in obj.values() for v in self._safe_extract_tensors(value)]
            if isinstance(obj, (list, tuple)):
                return [v for item in obj for v in self._safe_extract_tensors(item)]
            return []
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            return []

    def process_features(self, pt_path):
        """增强特征处理流程"""
        try:
            # 安全加载数据（启用weights_only）
            data = torch.load(pt_path, map_location='cpu', weights_only=True)
            
            # 提取基础特征
            raw_features = np.concatenate(self._safe_extract_tensors(data))
            if raw_features.size < 1:
                print(f"⚠️ {os.path.basename(pt_path)}: 无有效特征")
                return None

            # 动态降维策略
            n_samples, n_features = 1, raw_features.shape[0]
            valid_components = min(n_features, n_samples, self.max_components)
            
            if valid_components < 2:
                features = raw_features.reshape(1, -1)
            else:
                if not self.is_fitted:
                    self.pca.n_components = valid_components
                    self.pca.fit(raw_features.reshape(1, -1))
                    self.is_fitted = True
                features = self.pca.transform(raw_features.reshape(1, -1))

            # 标准化处理
            return self.scaler.fit_transform(features)
        except Exception as e:
            print(f"处理 {os.path.basename(pt_path)} 失败: {str(e)}")
            return None

# --------------------------
# 优化模型参数（调整版）
# --------------------------
def get_robust_model():
    """鲁棒性更强的模型参数"""
    return lgb.LGBMRegressor(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=5,  # 限制深度防止过拟合
        learning_rate=0.02,
        n_estimators=300,
        subsample=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        min_child_samples=10,
        importance_type='split'
    )

# --------------------------
# 增强预测流程（修复版）
# --------------------------
class DDGPredictor:
    def __init__(self):
        self.feature_engineer = FeatureEngineer(max_components=30)
        self.model = get_robust_model()
        self.cv_enabled = True
        
    def _safe_predict(self, features):
        """带异常捕获的预测"""
        try:
            if features is None or features.shape[1] == 0:
                return None
            if self.cv_enabled and features.shape[0] >= 5:
                return np.mean(cross_val_predict(self.model, features, cv=3, n_jobs=-1))
            return self.model.predict(features)[0]
        except Exception as e:
            print(f"预测失败: {str(e)}")
            return None

    def predict_ddg(self, pt_path):
        """鲁棒预测流程"""
        features = self.feature_engineer.process_features(pt_path)
        return self._safe_predict(features)

# --------------------------
# 增强验证模块
# --------------------------
def robust_validation(predictor, folder_path):
    """改进的验证方法"""
    valid_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.pt') and '_ΔΔG=' in f:
            try:
                float(f.split('_ΔΔG=')[1].split('.pt')[0])
                valid_files.append(f)
            except:
                continue
    
    y_pred, y_true = [], []
    for f in valid_files[:20]:  # 最多验证20个样本
        path = os.path.join(folder_path, f)
        pred = predictor.predict_ddg(path)
        if pred is not None:
            y_pred.append(pred)
            y_true.append(float(f.split('_ΔΔG=')[1].split('.pt')[0]))
    
    if len(y_pred) > 5:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"验证RMSE: {rmse:.3f} (基于{len(y_pred)}个样本)")
        return rmse
    print("⚠️ 验证样本不足")
    return None

# --------------------------
# 主执行流程（改进版）
# --------------------------
def robust_pipeline(input_folder, output_csv):
    predictor = DDGPredictor()
    
    print("=== 阶段1: 模型验证 ===")
    robust_validation(predictor, input_folder)
    
    print("\n=== 阶段2: 批量预测 ===")
    results = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.pt'):
            filepath = os.path.join(input_folder, filename)
            try:
                ddg = predictor.predict_ddg(filepath)
                if ddg is not None:
                    results.append({
                        'filename': filename,
                        'predicted_ddg': ddg,
                        'reliability': np.clip(1 - abs(ddg)/10, 0, 1)  # 更合理的置信度计算
                    })
            except Exception as e:
                print(f"处理失败 {filename}: {str(e)}")
    
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n✅ 完成! 结果保存至: {output_csv}")

if __name__ == "__main__":
    robust_pipeline(
        input_folder='s699/s699_out/',
        output_csv='robust_predictions.csv'
    )