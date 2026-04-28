### Project Setup and Install Required Packages

```bash
mkdir 01-data-pipeline-app
cd 01-data-pipeline-app
mkdir data src logs
touch main.py requirements.txt
```

```bash
mkdir src
touch src/config.py
touch src/validator.py
touch src/cleaner.py
touch src/transformer.py
touch src/pipeline.py
mkdir data
touch data/raw_data.csv
```

```bash
pip install -r requirements.txt
```

### Putting raw data in `data/raw_data.csv`

```bash
id,name,age,department,salary,date
1,John,28,IT,50000,2024-01-01
2,Alice,,HR,40000,2024-01-02
3,Bob,35,IT,,2024-01-03
4,Eve,29,Finance,60000,2024-01-04
5,Tom,40,HR,45000,
```

### Writing project code for project

1. Config (Central Control) - `src/config.py`
2. Validator - `src/validator.py`
3. Cleaner - `src/cleaner.py`
4. Transformer - `src/transformer.py`
5. Pipeline - `src/pipeline.py`
6. Main Entry - `main.py`

### Run the application

```bash
python main.py
```

> Should be create file `data/cleaned_transformed.csv`
