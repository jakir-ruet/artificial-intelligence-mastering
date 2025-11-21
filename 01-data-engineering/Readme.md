### Introduction to Data Engineering

Data Engineering for Machine Learning is the practice of collecting, storing, processing, and preparing data so that ML models can be trained, validated, deployed, and used effectively.

It focuses on building reliable, scalable data pipelines that deliver clean, consistent, and high-quality data for machine learning systems.

![Data Engineering](/img/data-eng.png)

#### The Key Stages In ML Data Engineering

- `Collection:` Gather data from APIs, logs, databases, etc.
- `Cleaning:` Remove duplicates, handle missing/nulls.
- `Preprocessing:`Normalize, encode, standardize.
- `Validation:` Check integrity, Data types, Ranges.

#### Common Tools Used In ML Data Engineering

- Python
- Pandas
- NumPy
- Ydata Profiling

#### Related terms

##### Raw Data Inspection

It is manual, initial look at the raw data exactly as it comes from the source.

> Purpose: To understand the structure, format, and content before any processing.

`Examples:`
- Opening a CSV to see columns and messy values
- Checking JSON logs for structure
- Viewing a few rows in a database table

#### Data Profiling

It is systematic, automated analysis that computes statistics and detects patterns in the data.

> Purpose: To measure data quality, distribution, and validity.

Outputs of data profiling:
- % nulls in each column
- min / max / mean / std
- value distributions (histogram)
- uniqueness, cardinality
- pattern frequency (email formats, phone formats)
- outliers

#### Raw Data Inspection vs Data Profiling

| Topic         | Raw Data Inspection                  | Data Profiling                               |
| ------------- | ------------------------------------ | -------------------------------------------- |
| **Type**      | Manual                               | Automated / programmatic                     |
| **Purpose**   | Understand the raw shape and content | Understand quality, statistics, and patterns |
| **Scale**     | Small datasets or sample rows        | Large datasets (millions of rows)            |
| **Output**    | Observations                         | Metrics, statistics, anomalies               |
| **Speed**     | Slow, depends on human               | Fast, scalable                               |
| **When Used** | First step                           | After ingestion or during pipeline QA        |

### Data Collection and Gathering

### Data Cleansing and Preprocessing

### Data Validation and Preparation for Training
