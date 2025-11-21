### Data Ingestion Approaches - Batch vs Real-Time

| **Category**   | **Batch Processing**                              | **Streaming Processing (Real-Time)**                       |
| -------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| **Use Case**   | Daily reports, periodic analytics, system backups | Sensor data, user interactions, real-time event processing |
| **Latency**    | High (minutes to hours)                           | Low (milliseconds to seconds)                              |
| **Complexity** | Simple setup with scheduled workflows             | Higher complexity with continuous data pipelines           |
| **Tools**      | Pandas, Apache NiFi, Airflow                      | Kafka, Spark Streaming, Flink                              |

### ETL (Extract, Transform, Load)

ETL is a data integration process where data is first extracted from sources, then transformed into a clean and usable format, and finally loaded into a target system such as a data warehouse. Transformations happen before the data is stored.

### ELT (Extract, Load, Transform)

ELT is a modern data integration process where data is extracted from sources, loaded directly into a storage system (like a data lake or cloud warehouse), and then transformed within that system. Transformations happen after loading.

#### ETL vs ELT – Comparison Table

| **Category**                | **ETL**                         | **ELT**                               |
| --------------------------- | ------------------------------- | ------------------------------------- |
| **Order of Steps**          | Extract → Transform → Load      | Extract → Load → Transform            |
| **Best For**                | Structured data                 | Semi-structured & unstructured data   |
| **Where Processing Occurs** | Before loading into storage     | After loading into storage (in-place) |
| **Performance**             | Limited by ETL server resources | Scales with cloud storage & compute   |
| **Typical Use Case**        | Traditional data warehouses     | Cloud data lakes, lakehouses          |

### Scalability Considerations and Challenges

- Data volume growth
- Automation and scheduling
- Error handling
- Tooling and integration
- Monitoring and performance

### Data Validation and Preparation for Training

- Multiple data sources
- Central ingestion layer
- Modular cleaning and transformation logic
- Unified output

![Data Validation and Preparation](/img/valid-prep-raining.png)

### Feature Engineering and Validation in Pipelines

- Transforming raw data into meaningful features
- Directly impacts model performance
- Converting timestamps to day of week/hour, one-hot encoding categories

#### Core Feature Engineering Techniques

- Normalization and Scaling
- Encoding Categorical Variables
- Handling Missing Data
- Basic Transformations
