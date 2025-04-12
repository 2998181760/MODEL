#!/usr/bin/env python
import argparse
import torch
import pandas as pd
import numpy as np
from glob import glob
import lightgbm as lgb
import logging
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_embeddings(embedding_dir):
    embedding_dict = {}
    for file in glob(f'{embedding_dir}/*.pt'):
        key = file.split('/')[-1].split('.')[0]
        embedding_dict[key] = torch.load(file)['mean_representations'][33].detach().numpy()
    return embedding_dict

def parse_mutation(df):
    df['wt'] = df['mutation'].str[0]
    df['mut'] = df['mutation'].str[-1]
    df['position'] = df['mutation'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    return df

def build_features(df, embedding_dict):
    """Build both WT-MUT and MUT1-MUT2 features."""
    X_raw, y_raw, ids_raw = [], [], []
    X_thermo, y_thermo, ids_thermo = [], [], []

    # WT - MUT
    for _, row in df.iterrows():
        wt_key = f"{row['pdb']}_{row['chain']}"
        mut_key = f"{row['pdb']}_{row['chain']}_{row['mutation']}"
        if wt_key in embedding_dict and mut_key in embedding_dict:
            vec = embedding_dict[wt_key] - embedding_dict[mut_key]
            X_raw.append(vec)
            y_raw.append(row['ddG'])
            ids_raw.append(f"{mut_key}_raw")

    # MUT1 - MUT2
    grouped = df.groupby(['pdb', 'chain', 'position', 'wt'])
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
                ids_thermo.append(f"{key1}_to_{row2['mutation']}_thermo")

    # 合并特征
    X = np.concatenate([X_raw, X_thermo], axis=0)
    y = np.concatenate([y_raw, y_thermo], axis=0)
    ids = ids_raw + ids_thermo

    return X, y, ids

def main():
    parser = argparse.ArgumentParser(description="Mixed feature prediction using LightGBM model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained LightGBM model")
    parser.add_argument("--output", type=str, required=True, help="Path to output prediction CSV")
    parser.add_argument("--data", type=str, default="../S669.tsv", help="Test dataset")
    parser.add_argument("--embedding_dir", type=str, default="esm2_650m_embeds", help="Directory with ESM embeddings")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info(f"Loading model from {args.model}")
    model = lgb.Booster(model_file=args.model)

    logger.info(f"Loading test data from {args.data}")
    df = pd.read_csv(args.data, sep='\t')
    df = parse_mutation(df)

    logger.info(f"Loading embeddings from {args.embedding_dir}")
    embedding_dict = load_embeddings(args.embedding_dir)

    logger.info("Building features...")
    X_test, y_test, ids = build_features(df, embedding_dict)

    logger.info(f"Predicting {len(X_test)} samples...")
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    pearson_corr, _ = pearsonr(y_test, preds)
    spearman_corr, _ = spearmanr(y_test, preds)

    logger.info("Evaluation metrics:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"Pearson: {pearson_corr:.4f}")
    logger.info(f"Spearman: {spearman_corr:.4f}")

    # Save results
    results_df = pd.DataFrame({
        "id": ids,
        "actual_ddG": y_test,
        "predicted_ddG": preds,
        "error": y_test - preds
    })
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
