### Apache Airflow

It's an open-source workflow orchestration platform for programmatically authoring, scheduling, and monitoring data pipelines. Let think an analogy, Airflow is like a `film director` — doesn't act, sing, or do cinematography, but tells everyone what to do, when, and in what order.

#### Apache Airflow Components

| Component              | Description                                            | Example                                                    |
| ---------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| **Scheduler**          | Triggers workflows based on schedule/dependencies      | `airflow scheduler` process; checks @daily DAG at midnight |
| **Executor**           | Defines how tasks run (local, distributed, Kubernetes) | `CeleryExecutor` distributes 100 tasks across 10 workers   |
| **Web Server**         | UI dashboard for monitoring & management               | `http://localhost:8080` → view DAGs, logs, failures        |
| **Metadata DB**        | Stores all state (DAGs, tasks, variables, history)     | PostgreSQL `task_instance` table with `state='success'`    |
| **Worker**             | Executes individual task code                          | Worker #3 runs `transform_data` task, reports back         |
| **DAG Directory**      | Folder where Airflow watches for Python DAG files      | `/airflow/dags/` containing `sales_etl.py`                 |
| **Message Queue**      | Buffer between scheduler & workers (CeleryExecutor)    | Redis holds queue: `[extract_api, transform, load]`        |
| **Task Instance**      | Single run of a task in a specific DAG run             | `extract_sql` task took 12s, state `success`               |
| **DAG Run**            | One full execution of a DAG for a time interval        | Jan 2 run processes data from Jan 1 interval               |
| **Operator**           | Template defining what a task does                     | `BashOperator(cmd='echo Hello')`                           |
| **Sensor**             | Waits for a condition before completing                | `FileSensor(filepath='/data.csv', wait=30s)`               |
| **Hook**               | Interface to external systems (DBs, cloud, APIs)       | `PostgresHook.run('SELECT * FROM users')`                  |
| **Connection**         | Stored credentials for external services               | `aws_prod`: S3, access key, secret, region                 |
| **Variable**           | Key-value config store accessible from DAGs            | `Variable.get('slack_url')` returns webhook                |
| **XCom**               | Small data exchange between tasks (max 1MB)            | Task A pushes `row_count=1000`, Task B pulls it            |
| **Pool**               | Limits concurrent tasks across DAGs                    | `db_writes` pool: max 3 simultaneous DB writes             |
| **Trigger Rule**       | Defines when task runs based on upstream states        | `all_done` runs even if upstream failed (cleanup)          |
| **SLA**                | Max expected task time; alerts if exceeded             | `sla=2hr` → email if task takes 3 hours                    |
| **Callback**           | Function triggered on task success/failure             | Send Slack alert when task fails                           |
| **Plugins**            | Custom operators, hooks, or UI extensions              | Custom `SalesforceOperator` from plugin folder             |
| **KubernetesExecutor** | Each task runs in its own ephemeral pod                | 10 tasks = 10 pods, each pod runs one task                 |

> Priority Legend

| Priority  | Components                                        |
| --------- | ------------------------------------------------- |
| Must know | DAG, Task, Operator, Scheduler, Executor          |
| Important | Web Server, Metadata DB, Sensor, Hook, Connection |
| Advanced  | XCom, Pool, Trigger Rule, SLA, Callback, Plugins  |

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
