# Mini-Project 2 (Inference: LightGBM + Logistic Regression)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training (run locally; do NOT train on the server)
- LightGBM (main model):
  ```bash
  python train_ml.py --train_path train.csv --config config/config.yaml --model_dir artifacts/ml
  ```
  Outputs: `artifacts/ml/preprocessor.pkl`, `artifacts/ml/lgbm_model.pkl`, `artifacts/ml/threshold.txt`.

- Logistic Regression (baseline for report):
  ```bash
  python train_logreg.py --train_path train.csv --config config/config.yaml --model_dir artifacts/logreg
  ```
  Outputs: `artifacts/logreg/preprocessor_logreg.pkl`, `logreg_model.pkl`, `logreg_best_threshold.txt`.

## Inference (server command)
TAs will run under `miniproject2/`:
```bash
./inference.sh --test_path /service/res/miniproject2/test.csv
```
This activates venv, loads artifacts in `artifacts/ml`, and writes `predictions.csv` to `miniproject2/`.

## Project Structure
- inference.sh
- README.md
- inference.py
- train_ml.py
- train_logreg.py
- config/config.yaml
- artifacts/ (trained models & preprocessor)
- src/ (preprocessing, features, models, utils)

## Notes
- Do not upload any `.csv` data files. Keep only code, models, and artifacts.
- Server is CPU-only; ensure inference finishes within time limits.
- Keep folder name `miniproject2/` directly under `/service/home/<studentID>/` on the server.
