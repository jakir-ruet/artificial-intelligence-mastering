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

#### Common Methods of Data Collection

- `Databases` > SQL and NoSQL databases.
- `APIs` > REST data
- `Flat Files` > CSV, JSON and XML Logs files
- `Streaming Data` > Sensors, Kafka and Webbooks.

#### Batch Ingestion

Batch ingestion is a data ingestion method where data is collected over a period of time (e.g., hourly, daily) and then processed all at once in bulk.
It’s useful for large volumes of data that don’t need immediate processing.

#### Real-time Ingestion

Real-time ingestion is a method where data is continuously ingested and processed the moment it arrives. It’s used when low latency is critical, such as fraud detection or live dashboards.

#### Batch vs. Real-time Ingestion

| Aspect           | **Batch Ingestion**                            | **Real-time Ingestion**                             |
| ---------------- | ---------------------------------------------- | --------------------------------------------------- |
| **How it works** | Runs on a **schedule** (hourly, daily, weekly) | Ingests data **continuously** as events happen      |
| **Latency**      | High latency (minutes to hours)                | Very low latency (milliseconds to seconds)          |
| **Use Cases**    | Reporting, analytics, data warehousing         | Fraud detection, recommendation systems, monitoring |
| **Data Volume**  | Handles **large volumes** at once              | Handles **continuous streams** of smaller events    |
| **Complexity**   | Easier to design, implement, and debug         | More complex; requires real-time pipelines          |
| **Tools**        | Airflow, AWS Glue, Spark Batch, cron jobs      | Kafka, Kinesis, Flink, Spark Streaming              |
| **Architecture** | Simple pipelines, fewer moving parts           | Requires scalable, fault-tolerant architecture      |
| **Cost**         | Usually cheaper                                | Often more expensive due to always-on systems       |

### Data Cleaning and Preprocessing

#### Why Data Cleaning and Preprocessing Matters

Data cleaning and preprocessing matter because they:

- Improve accuracy
- Reduce noise
- Prevent bias
- Help models train faster
- Ensure reliable decision-making

#### Key Techniques For Cleaning Data

- Handling Missing Values `(Drop rows/Columns)`
- Removing Duplications `(.drop_duplicates())`
- Normalization vs Standardization
  - Normalization: [0,1] scale (e.g., MinMaxScaler)
  - Standardization: mean=0, std=1 (e.g., z-score)

### Data Validation and Preparation for Training

#### Why Data Validation Matters

- Data validation ensures input data meets the required quality before model training
- Reduces model errors, overfitting, and unexpected behavior

Helps catch:
- Incorrect types (e.g., strings in numerical fields)
- Invalid/missing values
- Out-of-range or inconsistent data

Validation Types:
- Type checking
- Range validation
- Null checks and handling
- Unique vs. duplicate value checks
- Value distribution anomalies

### Training Data Formats and Export

| **Format**  | **Structure**        | **Type**                     | **Use Case**                                | **Pros**                                                    | **Cons**                                                    |
| ----------- | -------------------- | ---------------------------- | ------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| **CSV**     | Flat, tabular        | Tabular                      | Classic ML tasks, spreadsheets              | Simple, human-readable, universally supported               | Not suitable for nested data; no data types; can be large   |
| **JSON**    | Nested, hierarchical | Structured / semi-structured | NLP, APIs, event logs, configurations       | Flexible, supports complex structures                       | Harder to read, more verbose, inconsistent schemas possible |
| **Parquet** | Columnar, binary     | Big data / tabular           | Large-scale ML pipelines, Spark, data lakes | Highly compressed, fast read/write, optimized for analytics | Not human-readable; tooling needed                          |

### Environment Setup

```bash
python3 -m venv venv # Install virtual environment
source venv/bin/activate
```

### Install Pandas

```bash
pip install --upgrade pip
pip install pandas
```

> Tip: If you plan to use NumPy, Matplotlib, or Scikit-learn:

```bash
pip install numpy matplotlib scikit-learn
```

### Ydata profile install

```bash
pip list | grep ydata-profiling
pip install ydata-profiling
```

### Jupyter install

```bash
pip install notebook
jupyter notebook
```

### Deactivate env

```bash
deactivate
```
