## Prerequisites

1. Programming Skills
	- Comfortable programming in Python
	- Ability to write scripts and functions (not only notebooks)
2. Machine Learning
	- Understanding of ML concepts
	- Experience with scikit-learn
	- Familiar with:
		- classification
		- regression
		- train/test split
		- evaluation metrics
3. Databricks Knowledge
	- Familiar with Databricks platform basics
	- Prerequisite Courses
	- Before MLflow course, recommended:
	1. Machine Learning with scikit-learn
		- Building ML models in Python
		- Data preprocessing
		- Model training and evaluation
	2. Databricks Fundamentals
		- Getting Started with Databricks Lakehouse Platform
		- Working in notebooks and clusters
	3. Apache Spark on Databricks
		- Data processing with Spark
		- Distributed computing basics

### ML Flow

MLflow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It allows data scientists and engineers to track experiments, package code into reproducible runs, and manage model versions from development to production.

**Core Components**
1. `Tracking:` Log parameters, code versions, metrics, and output files for later visualization and comparison.
2. `Projects:` Package code in a standardized format to ensure it runs the same way on any platform.
3. `Models:` A standard format for packaging models that can be used in diverse serving environments like Amazon SageMaker or Microsoft Azure ML.
4. `Model Registry:` A central store to collaboratively manage model versions, stage transitions, and annotations.

### Databricks

Databricks is a unified cloud-based data and AI platform that simplifies big data processing, data engineering, and machine learning. It was founded by the original creators of **Apache Spark** and **MLflow**, which is why those tools are so tightly integrated into the platform.

**The `Data Lakehouse` Architecture**

Databricks pioneered the Data Lakehouse, a hybrid model that combines the best parts of two worlds:

- `Data Lakes:` The low cost and flexibility of storing unstructured data (images, videos, logs) in cloud storage like Amazon S3 or Azure Data Lake.
- `Data Warehouses:` The reliability, security, and performance of structured SQL databases.

**Key Capabilities**
1. `Data Engineering:` Build automated pipelines (ETL/ELT) using Delta Live Tables to clean and transform data at scale.
2. `SQL Analytics:` Use Databricks SQL to run high-performance queries and build dashboards directly on your lakehouse, similar to Snowflake or BigQuery.
3. `Machine Learning:` A dedicated workspace with managed MLflow for experiment tracking, plus Mosaic AI for training and serving large language models (LLMs).
4. `Governance:` Unity Catalog provides a single layer to manage permissions, auditing, and data lineage across all your data and AI assets.

### Pipeline Flow

```bash
Data Sources → Data Preparation → Feature Store → Model Training → Experiment Tracking (MLflow) → Model Registry → Deployment → Inference
```

### End-to-End ML Workflow (Databricks + MLflow Style)

| Step | Stage                        | Description                               | Key Components                             | Purpose                                             |
| ---- | ---------------------------- | ----------------------------------------- | ------------------------------------------ | --------------------------------------------------- |
| 1    | Data Preparation             | Collect and prepare raw data for ML       | Databases, APIs, Delta Tables              | Convert raw data into usable ML input               |
|      | Delta Tables                 | Versioned and reliable data storage layer | ACID transactions, time travel, versioning | Ensure data reliability and reproducibility         |
| 2    | Feature Store                | Central storage for ML features           | Feature tables, reusable feature pipelines | Ensure consistent features for training & inference |
| 3    | Model Training               | Train ML models using data                | scikit-learn, Spark ML, AutoML             | Build predictive models                             |
| 4    | Experiment Tracking (MLflow) | Track all ML experiments                  | MLflow Tracking                            | Logs parameters, metrics, models, artifacts         |
|      | Logged Data                  | What is stored per experiment             | params, metrics, model files, artifacts    | Enable comparison & reproducibility                 |
| 5    | Model Registry               | Central model management system           | MLflow Model Registry                      | Version control and lifecycle management            |
|      | Model Stages                 | Lifecycle stages of models                | Staging, Production, Archived              | Control deployment readiness                        |
| 6    | Model Inference              | Use trained model for predictions         | Batch jobs, Streaming pipelines, REST APIs | Deliver predictions to applications                 |
