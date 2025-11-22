# End-to-End Data Pipeline

This module demonstrates a complete end-to-end data processing pipeline with modular design, data validation, and automated output management.

## Files

- `Pipeline.py:` Main pipeline implementation
- `output/:` Directory containing timestamped pipeline outputs
  - `titanic-cleaned-*.csv:` Multiple timestamped versions of processed data
- `index.html:` Generated data profiling report

## What You'll Learn

- End-to-end pipeline design and implementation
- Modular function-based architecture
- Data validation and quality assessment
- Automated output management with timestamps
- Pipeline orchestration and workflow management

## Key Concepts

### Pipeline Architecture

- `Modular Design:` Separate functions for each pipeline stage
- `Function Composition:` Building complex workflows from simple functions
- `Error Handling:` Robust pipeline execution
- `Output Management:` Timestamped file generation

### Data Processing Stages

1. `Data Loading:` Importing datasets from various sources
2. `Data Cleaning:` Handling missing values and data quality issues
3. `Feature Engineering:` Creating new features and transformations
4. `Data Validation:` Quality assessment and profiling
5. `Data Export:` Saving processed data with version control

## Usage

```bash
python Pipeline.py
```

## Pipeline Workflow

### 1. Data Loading (`load_data()`)

- Load Titanic dataset from seaborn
- Initial data inspection and validation

### 2. Data Cleaning (`clean_data()`)

- Handle missing values in 'embarked' column
- Impute missing ages with median values
- Fill missing embark_town with 'Unknown'

### 3. Feature Engineering (`engineer_features()`)

- Create age groups using binning
- Encode sex as binary variable
- Generate new categorical features

### 4. Data Validation (`validate_data()`)

- Generate comprehensive data profile
- Create HTML report for manual review
- Assess data quality metrics

### 5. Data Export (`export_data()`)

- Save processed data with timestamps
- Version control for pipeline outputs
- Organized output directory structure

## Pipeline Functions

### Data Loading

```python
def load_data():
    df = sns.load_dataset("titanic")
    return df
```

### Data Cleaning

```python
def clean_data(df):
    df = df.dropna(subset=["embarked"]).copy()
    df.loc[:, "age"] = df["age"].fillna(df["age"].median())
    df.loc[:, "embark_town"] = df["embark_town"].fillna("Unknown")
    return df
```

### Feature Engineering

```python
def engineer_features(df):
    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 120],
                             labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
    df["sex_encoded"] = df["sex"].map({"male": 0, "female": 1})
    return df
```

## Output Management

### Timestamped Files

- Automatic timestamp generation: `YYYYMMDD_HHMMSS`
- Version control for pipeline outputs
- Organized output directory structure

### Generated Reports

- `HTML Profiling Report:` Comprehensive data quality assessment
- `CSV Output:` Processed data ready for ML models
- `Version History:` Multiple pipeline runs preserved

## Dependencies

- `pandas:` Data manipulation and analysis
- `seaborn:` Dataset loading and visualization
- `ydata_profiling:` Data quality assessment
- `datetime:` Timestamp generation
- `os:` File system operations

## Learning Outcomes

After completing this module, you'll understand:
- How to design end-to-end data pipelines
- Modular pipeline architecture
- Data validation and quality assessment
- Automated output management
- Pipeline orchestration techniques
- Best practices for production data pipelines
- Version control for data processing workflows
