## Environment Setup

```bash
python3 -m venv venv # Install virtual environment
source venv/bin/activate
```

```bash
pip3 install virtualenv
```

```bash
virtualenv venv
```

```bash
source venv/bin/activate
```

## Install Pandas

```bash
pip install --upgrade pip
pip install pandas
```

> Tip: If you plan to use NumPy, Matplotlib, or Scikit-learn:

```bash
pip install numpy matplotlib scikit-learn
```

## Ydata profile install

```bash
pip list | grep ydata-profiling
pip install ydata-profiling
```

## Jupyter install

```bash
pip install notebook
jupyter notebook
```

## Deactivate env

```bash
deactivate
```

## Files

- `ingest-csv.py`: Demonstrates CSV data loading and basic analysis
- `ingest-api.py`: Shows how to fetch data from REST APIs
- `hris.csv`: Sample HRIS (Human Resources Information System) dataset
