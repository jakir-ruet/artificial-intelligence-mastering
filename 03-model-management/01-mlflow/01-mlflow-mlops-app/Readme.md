### Architecture (Real MLOps Flow)

```bash
Raw Data (CSV/API)
        ↓
Data Cleaning (Pandas)
        ↓
Feature Engineering
        ↓
ML Training (scikit-learn)
        ↓
MLflow Tracking (Experiments)
        ↓
Model Registry (Versioning)
        ↓
FastAPI Service
        ↓
Docker Container
        ↓
Prediction API (Production)
```

### App Structure

```bash
mlflow-mlops-project/
│
├── data/
│   └── housing.csv
│
├── src/
│   ├── train.py
│   ├── preprocess.py
│   ├── model.py
│
├── app/
│   └── main.py   # FastAPI
│
├── mlruns/       # MLflow local tracking
├── requirements.txt
├── Dockerfile
└── README.md
```

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
