### Apache Airflow

It's an open-source workflow orchestration platform for programmatically authoring, scheduling, and monitoring data pipelines. Let think an analogy, Airflow is like a `film director` — doesn't act, sing, or do cinematography, but tells everyone what to do, when, and in what order.

### Apache Airflow Architecture

![Apache Airflow Architecture](/img/airflow-architecture.png)

#### Components

##### Control Layer

| Component   | Short Description                                                      | Example                                                                                            |
| ----------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| DAGs Folder | Directory containing Python files that define workflows                | `/opt/airflow/dags/` with `sales_etl.py`, `ml_pipeline.py`                                         |
| Webserver   | Web UI to monitor, trigger, and debug workflows                        | Visit `http://localhost:8080` → see DAG `daily_sales` failed → click to view logs                  |
| Scheduler   | Brain that parses DAGs and triggers tasks when dependencies are met    | At 2:00 AM, scheduler sees `@daily` DAG → creates DAG Run → queues `extract_task`                  |
| Executor    | Dispatcher that decides **how** tasks run (local, distributed, or K8s) | `CeleryExecutor` sends 50 tasks to 5 worker machines; `LocalExecutor` runs 4 tasks on same machine |

##### Execution Layer

| Component | Short Description                                   | Example                                                                                      |
| --------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Worker    | Process that actually executes a single task's code | Worker #3 runs `PythonOperator` → executes `transform_data()` function → writes result to S3 |

##### Storage Layer

| Component         | Short Description                                                             | Example                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Message Queue     | Buffer that holds tasks waiting for workers (used with CeleryExecutor)        | Redis queue contains: `[extract_api, transform, load]` → worker picks next task when idle                 |
| Metadata Database | Central database storing all state: task status, DAGs, variables, connections | PostgreSQL `task_instance` table: `dag_id='sales'`, `task_id='load'`, `state='success'`, `duration=12.3s` |

##### Connection Components (Inside Metadata DB)

| Component  | Short Description                                    | Example                                                               |
| ---------- | ---------------------------------------------------- | --------------------------------------------------------------------- |
| Connection | Stored credentials for external services (encrypted) | Connection `aws_prod`: `S3`, `AKIA...`, `secret`, `region=us-east-1`  |
| Variable   | Key-value store for runtime configuration            | `Variable.get('slack_webhook')` returns `https://hooks.slack.com/xxx` |

##### Integration Components

| Component | Short Description                                  | Example                                                            |
| --------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| Hook      | Interface that wraps connection to external system | `S3Hook(aws_conn_id='aws_prod').download_file(key='data.csv')`     |
| Operator  | Template that defines a single task's work         | `PythonOperator(task_id='clean', python_callable=clean_data)`      |
| Sensor    | Special operator that waits for a condition        | `S3KeySensor(bucket='incoming', key='file.csv', poke_interval=30)` |

##### External Systems (Orchestrated, Not Part of Airflow)

| System     | Short Description                  | Example                                                                  |
| ---------- | ---------------------------------- | ------------------------------------------------------------------------ |
| Snowflake  | Cloud data warehouse               | `SnowflakeOperator(sql="COPY INTO sales FROM @s3_stage")`                |
| Redshift   | AWS data warehouse                 | `RedshiftSQLOperator(sql="INSERT INTO analytics SELECT * FROM staging")` |
| BigQuery   | Google's serverless data warehouse | `BigQueryInsertJobOperator(config={"query": "SELECT * FROM table"})`     |
| dbt        | Data transformation tool           | `DbtRunOperator(project_dir='/dbt/models')`                              |
| Databricks | Unified analytics platform         | `DatabricksSubmitRunOperator(json={"notebook_task": {...}})`             |
| Kubernetes | Container orchestration            | `KubernetesPodOperator(image='spark:latest', cmds=['spark-submit'])`     |
| PostgreSQL | Relational database                | `PostgresOperator(sql="UPDATE users SET active=True")`                   |
| Tableau    | BI visualization                   | `TableauRefreshWorkbookOperator(workbook_id='123')`                      |
| S3         | Object storage                     | `S3Hook.load_file(filename='data.csv', bucket='my-bucket')`              |

> Priority Legend

| Priority  | Components                                        |
| --------- | ------------------------------------------------- |
| Must know | DAG, Task, Operator, Scheduler, Executor          |
| Important | Web Server, Metadata DB, Sensor, Hook, Connection |
| Advanced  | XCom, Pool, Trigger Rule, SLA, Callback, Plugins  |

### Airflow Install & Configuration

#### System Update & Upgrade

```bash
sudo apt update && sudo apt upgrade -y
```

#### Required Dependancy Install

```bash
sudo apt install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    curl \
    git
```

#### Create User and Password

```bash
sudo useradd -m -s /bin/bash airflow
sudo passwd airflow
```

```bash
sudo su - airflow
```

#### Environment Setup and Update

```bash
python3 -m venv airflow-venv
source airflow-venv/bin/activate
```

```bash
pip install --upgrade pip setuptools wheel
```

#### Set Airflow version

```bash
export AIRFLOW_VERSION=3.2.0
export PYTHON_VERSION=3.14
export CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

#### Install Apache Airflow

```bash
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
pip install apache-airflow==3.2.0
```

#### Initialize Airflow database

```bash
export AIRFLOW_HOME=~/airflow
airflow db migrate
```

#### Create admin user

```bash
airflow standalone # Recommended
```

**Or**

```bash
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

#### Start Airflow services - `Terminal 1`

```bash
airflow webserver --port 8085
```

#### Start Airflow services - Terminal 2 (Scheduler)

```bash
su - airflow
python3 -m venv airflow-venv
source airflow-venv/bin/activate
airflow scheduler
```

```bash
http://<your-server-ip>:8080 # Username: admin, Password: collect from cli log
```
