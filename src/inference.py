import argparse
import os
import joblib
import numpy as np
import pandas as pd

from utils import smape

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--output_path', type=str, default='predictions.csv')
    parser.add_argument('--model_path', type=str, default='models/model.joblib')
    args = parser.parse_args()

    assert os.path.exists(args.model_path), f"Model not found: {args.model_path}"
    bundle = joblib.load(args.model_path)
    pipe = bundle['pipeline']
    cfg = bundle.get('config', {})
    clip_min = float(cfg.get('target', {}).get('clip_min', 0.0))

    df_test = pd.read_csv(args.test_path)
    assert 'name' in df_test.columns, 'test.csv must have column: name'
    names = df_test['name'].astype(str)
    preds = pipe.predict(names)
    preds = np.asarray(preds, dtype=float)
    preds = np.maximum(preds, clip_min if clip_min else 0.01)
    out = pd.DataFrame({'name': names, 'price': preds})
    out.to_csv(args.output_path, index=False)
    print(f"Wrote predictions to {args.output_path} (shape={out.shape})")

if __name__ == '__main__':
    main()
