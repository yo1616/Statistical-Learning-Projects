import argparse
import os
import time
import yaml
import joblib
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from preprocess import normalize_texts
from pipeline_utils import (
    wrap_to_df,
    simple_numeric_feats,
    clip_numeric_feats,
    numeric_to_array,
)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    out = np.zeros_like(denom, dtype=float)
    out[mask] = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return 200.0 * np.mean(out)


def build_pipeline(cfg, seed: int):
    vec_cfg = cfg.get('vectorizer', {})
    mdl_cfg = cfg.get('model', {})
    analyzer = vec_cfg.get('analyzer', 'char')
    ngram_range = tuple(vec_cfg.get('ngram_range', [1, 5]))
    kind = vec_cfg.get('kind', 'hashing')
    if kind == 'hashing':
        text_vec = HashingVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            n_features=int(vec_cfg.get('n_features', 2**20)),
            alternate_sign=bool(vec_cfg.get('alternate_sign', False)),
            norm='l2',
            lowercase=True,
        )
    elif kind == 'tfidf':
        text_vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=int(vec_cfg.get('max_features', 400_000)),
            min_df=int(vec_cfg.get('min_df', 2)),
            lowercase=True,
            sublinear_tf=True,
        )
    else:
        raise ValueError(f"unknown vectorizer kind {kind}")
    text_branch = Pipeline(
        [
            ('norm', FunctionTransformer(normalize_texts, validate=False)),
            ('vec', text_vec),
        ]
    )
    num_branch = Pipeline(
        [
            ('num', FunctionTransformer(simple_numeric_feats, validate=False)),
            ('clip', FunctionTransformer(clip_numeric_feats, validate=False)),
            ('to_array', FunctionTransformer(numeric_to_array, validate=False)),
            ('scaler', StandardScaler(with_mean=False)),
        ]
    )
    ct = ColumnTransformer(
        transformers=[
            ('text', text_branch, 'name'),
            ('num', num_branch, 'name'),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    name = mdl_cfg.get('name', 'ridge')
    params = mdl_cfg.get('params', {}) or {}
    if name == 'ridge':
        base = Ridge(alpha=float(params.get('alpha', 1.0)))
    elif name == 'elasticnet':
        base = ElasticNet(
            alpha=float(params.get('alpha', 0.1)),
            l1_ratio=float(params.get('l1_ratio', 0.2)),
            max_iter=int(params.get('max_iter', 2000)),
            random_state=seed,
        )
    elif name == 'linear_svr':
        base = LinearSVR(
            C=float(params.get('C', 1.0)),
            epsilon=float(params.get('epsilon', 0.1)),
            max_iter=int(params.get('max_iter', 2000)),
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model {name}")
    model = TransformedTargetRegressor(
        regressor=base,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    pipe = Pipeline(
        [
            ('wrap', FunctionTransformer(wrap_to_df, validate=False)),
            ('feats', ct),
            ('model', model),
        ]
    )
    return pipe


DEFAULT_CFG = {
    'seed': 42,
    'cv_folds': 5,
    'target': {'clip_min': 0.0},
    'vectorizer': {
        'kind': 'tfidf',
        'analyzer': 'char',
        'ngram_range': [1, 5],
        'max_features': 400_000,
        'min_df': 2,
    },
    'model': {
        'name': 'elasticnet',
        'params': {
            'alpha': 0.05,
            'l1_ratio': 0.3,
            'max_iter': 4000,
        },
    },
}


DEFAULT_SEARCH_SPACE = {
    'hashing_ridge': {
        'feats__text__vec__ngram_range': [(1, 3), (1, 4), (1, 5)],
        'feats__text__vec__n_features': [2**18, 2**19, 2**20],
        'model__regressor__alpha': [0.3, 0.5, 1.0, 2.0, 3.0],
    },
    'tfidf_elasticnet': {
        'feats__text__vec__ngram_range': [(1, 3), (1, 4), (1, 5)],
        'feats__text__vec__max_features': [200_000, 300_000, 400_000],
        'model__regressor__alpha': [0.01, 0.05, 0.1, 0.2],
        'model__regressor__l1_ratio': [0.1, 0.2, 0.4, 0.6],
    },
    'tfidf_linear_svr': {
        'feats__text__vec__ngram_range': [(1, 3), (1, 4), (1, 5)],
        'feats__text__vec__max_features': [200_000, 300_000, 400_000],
        'model__regressor__C': [0.5, 1.0, 2.0],
        'model__regressor__epsilon': [0.05, 0.1, 0.2],
    },
}


def load_config(path: str):
    data = {}
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    merged = {
        key: DEFAULT_CFG[key]
        for key in DEFAULT_CFG
    }
    merged['vectorizer'] = DEFAULT_CFG['vectorizer'].copy()
    merged['model'] = {
        'name': DEFAULT_CFG['model']['name'],
        'params': DEFAULT_CFG['model']['params'].copy(),
    }
    merged['target'] = DEFAULT_CFG['target'].copy()
    for key, value in data.items():
        if key not in ('vectorizer', 'model', 'target'):
            merged[key] = value
    if 'vectorizer' in data:
        merged['vectorizer'].update(data['vectorizer'] or {})
    if 'model' in data:
        model_block = data['model'] or {}
        if 'name' in model_block:
            merged['model']['name'] = model_block['name']
        if 'params' in model_block:
            merged['model']['params'].update(model_block['params'] or {})
    if 'target' in data:
        merged['target'].update(data['target'] or {})
    return merged


def load_search_space(path: str):
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return data
    return DEFAULT_SEARCH_SPACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--out_model_path', type=str, default='models/model.joblib')
    parser.add_argument('--cv', type=int, default=None)
    parser.add_argument('--search', type=str, default='none', choices=['none', 'random'])
    parser.add_argument('--n_iter', type=int, default=15)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.model == 'hashing_ridge':
        cfg['vectorizer']['kind'] = 'hashing'
        cfg['vectorizer']['analyzer'] = 'char'
        cfg['vectorizer']['ngram_range'] = [1, 5]
        cfg['model']['name'] = 'ridge'
        cfg['model']['params']['alpha'] = 0.5
    
    seed = args.seed if args.seed is not None else int(cfg.get('seed', 42))
    set_seed(seed)
    print('Load data...')
    df = pd.read_csv(args.train_path)
    if 'name' not in df.columns or 'price' not in df.columns:
        raise ValueError('train.csv must include name and price columns')
    X = df['name'].astype(str)
    y = df['price'].astype(float).to_numpy()
    n_folds = args.cv if args.cv is not None else int(cfg.get('cv_folds', 5))
    n_folds = max(2, min(n_folds, len(df)))
    print('Build pipeline...')
    pipe = build_pipeline(cfg, seed)
    scorer = make_scorer(smape, greater_is_better=False)
    if args.search == 'random':
        print(f'Random search ({args.n_iter} iters)...')
        key = args.model if args.model else f"{cfg['vectorizer']['kind']}_{cfg['model']['name']}"
        space = load_search_space(os.path.join('configs', 'search_space.yaml'))
        params_space = space.get(key, {})
        param_distributions = {}
        for param_key, param_values in params_space.items():
            if isinstance(param_values, list) and param_values and isinstance(param_values[0], list):
                param_distributions[param_key] = [tuple(x) for x in param_values]
            else:
                param_distributions[param_key] = param_values
        if not param_distributions:
            raise ValueError(f'no search space defined for key {key}')
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            scoring=scorer,
            cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed),
            random_state=seed,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X, y)
        print(f"Best CV SMAPE: {-search.best_score_:.4f}")
        print(f"Best Params: {search.best_params_}")
        pipe = search.best_estimator_
    else:
        print(f'CV {n_folds}-fold evaluation with SMAPE...')
        scores = -cross_val_score(
            pipe,
            X,
            y,
            scoring=scorer,
            cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed),
            n_jobs=-1,
        )
        print(f"CV SMAPE (per fold): {np.round(scores, 4)}")
        print(f"Mean CV SMAPE: {scores.mean():.4f} +/- {scores.std():.4f}")
        pipe.fit(X, y)
    dirpath = os.path.dirname(args.out_model_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    print('Save model...')
    joblib.dump(
        {
            'pipeline': pipe,
            'config': cfg,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': seed,
        },
        args.out_model_path,
    )
    print(f"Model saved to {args.out_model_path}")


if __name__ == '__main__':
    main()
