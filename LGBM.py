import os
import torch
import lightgbm as lgb
import numpy as np
import pandas as pd
from collections import OrderedDict

def load_model(model_path):
    """加载训练好的LightGBM模型"""
    return lgb.Booster(model_file=model_path)

def load_pt_features(pt_path):
    """通用.pt文件特征解析方法"""
    try:
        data = torch.load(pt_path, map_location=torch.device('cpu'))
        
        # 深度解析特征结构
        def extract_tensor(obj):
            """递归提取所有张量数据"""
            if isinstance(obj, torch.Tensor):
                return obj.numpy()
            elif isinstance(obj, (dict, OrderedDict)):
                # 处理新版重建格式
                if 'storage' in obj:
                    storage = obj['storage']._untyped_storage
                    rebuild_args = (
                        storage,
                        obj.get('storage_offset', 0),
                        obj.get('size', []),
                        obj.get('stride', []),
                        obj.get('requires_grad', False),
                        obj.get('backward_hooks', OrderedDict())
                    )
                    tensor = torch._utils._rebuild_tensor_v2(*rebuild_args)
                    return tensor.numpy()
                # 处理旧版存储格式
                return np.concatenate([extract_tensor(v) for v in obj.values()])
            elif isinstance(obj, (list, tuple)):
                return np.concatenate([extract_tensor(x) for x in obj])
            else:
                raise ValueError(f"Unsupported data type: {type(obj)}")

        # 从不同层级提取特征
        tensor_data = data.get('mean_representations', data.get('data', data))
        features = extract_tensor(tensor_data).flatten()
        
        if features.size == 0:
            raise ValueError("Empty feature array")
        return features
        
    except Exception as e:
        raise ValueError(f"解析失败: {str(e)}") from e

def predict_folder(folder_path, model_path, output_csv="predictions.csv"):
    """鲁棒的批量预测流程"""
    model = load_model(model_path)
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pt"):
            filepath = os.path.join(folder_path, filename)
            try:
                features = load_pt_features(filepath)
                
                # 维度验证
                if features.ndim != 1:
                    features = features.reshape(-1)  # 强制展平
                    print(f"警告: {filename} 特征已展平")
                    
                pred = model.predict(features.reshape(1, -1))[0]
                results.append((filename, pred))
                print(f"成功处理: {filename}")
                
            except Exception as e:
                print(f"失败文件: {filename}\n错误详情: {str(e)}\n{'━'*50}")
                results.append((filename, np.nan))
    
    # 保存结果
    pd.DataFrame(results, columns=["File", "ΔΔG"]).to_csv(output_csv, index=False)
    print(f"\n最终报告: 成功{len(results)-pd.isna(results).sum()}个 | 失败{pd.isna(results).sum()}个")

if __name__ == "__main__":
    predict_folder(
        folder_path="s699/s669_out/",
        model_path="lgbm_model.txt",
        output_csv="final_predictions.csv"
    )