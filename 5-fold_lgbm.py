import os
import torch
import lightgbm as lgb
import pandas as pd
from collections.abc import Mapping, Iterable

def load_lgb_model(model_path):
    """加载LightGBM模型"""
    return lgb.Booster(model_file=model_path)

def extract_tensors(data):
    """深度递归搜索数据结构中的张量"""
    if isinstance(data, torch.Tensor):
        return data
    
    # 处理字典类数据结构
    if isinstance(data, Mapping):
        # 优先检查常见字段名
        for key in ['features', 'data', 'input', 'x', 'representations', 
                   'mean_representations', 'hidden_states']:
            if key in data:
                result = extract_tensors(data[key])
                if result is not None:
                    return result
        # 深度遍历所有值
        for v in data.values():
            result = extract_tensors(v)
            if result is not None:
                return result
    
    # 处理列表/元组等可迭代对象
    elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        for item in data:
            result = extract_tensors(item)
            if result is not None:
                return result
    
    # 处理带有缓冲区的特殊对象
    if hasattr(data, '__dict__'):
        return extract_tensors(data.__dict__)
    
    return None

def load_features(file_path):
    """加载并解析.pt文件特征"""
    try:
        data = torch.load(file_path, map_location='cpu')
        
        # 递归提取张量
        tensor = extract_tensors(data)
        if tensor is not None:
            return tensor.numpy().reshape(1, -1)
        
        # 处理特殊包装类型
        if hasattr(data, 'cpu'):
            return data.cpu().numpy().reshape(1, -1)
        
        # 最终检查数据的存储结构
        if hasattr(data, 'storage') and isinstance(data.storage(), torch.Storage):
            return torch.tensor(data.storage()).numpy().reshape(1, -1)
            
        raise ValueError(f"无法解析的文件结构: {type(data)}")
        
    except Exception as e:
        raise ValueError(f"文件加载失败: {str(e)}") from e

def predict_ddg(model, features):
    """使用模型进行预测"""
    return model.predict(features)[0]

def main(model_path, input_folder, output_csv):
    """主处理流程"""
    model = load_lgb_model(model_path)
    results = []
    error_log = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pt'):
            file_path = os.path.join(input_folder, file_name)
            try:
                features = load_features(file_path)
                ddg_prediction = predict_ddg(model, features)
                results.append({
                    'File': file_name, 
                    'DDG_Prediction': ddg_prediction
                })
            except Exception as e:
                error_msg = f"Error processing {file_name}: {str(e)}"
                print(error_msg)
                error_log.append(error_msg)
    
    # 保存结果
    pd.DataFrame(results).to_csv(output_csv, index=False)
    
    # 保存错误日志
    if error_log:
        with open('feature_errors.log', 'w') as f:
            f.write('\n'.join(error_log))
    
    print(f"处理完成 | 成功: {len(results)} | 失败: {len(error_log)}")

if __name__ == '__main__':
    main(
        model_path='lgb_model_3.txt',
        input_folder='s699/s669_out/',
        output_csv='ddg_predictions2.csv'
    )