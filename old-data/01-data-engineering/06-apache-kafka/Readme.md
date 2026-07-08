### Apache Kafka

Apache Kafka is an open-source distributed event streaming platform used by thousands of companies for high-performance data pipelines, streaming analytics, data integration, and mission-critical applications.

#### Apache Kafka  vs RabbitMQ / ActiveMQ (Traditional MQ)

Comparing Apache Kafka with traditional messaging systems (like RabbitMQ or ActiveMQ) is less about “which is better” and more about architecture philosophy. They solve different problems, even though both move messages.

| Aspect                 | Apache Kafka                                               | RabbitMQ / ActiveMQ (Traditional MQ)                  |
| ---------------------- | ---------------------------------------------------------- | ----------------------------------------------------- |
| **Architecture Model** | Distributed **log-based streaming platform**               | **Queue-based message broker**                        |
| **Message Storage**    | Persistent (disk-based, append-only log)                   | Typically transient (deleted after consumption)       |
| **Message Retention**  | Configurable (time/size-based)                             | Usually none after acknowledgment                     |
| **Replay Capability**  | Yes (re-read using offsets)                                | No (once consumed, gone)                              |
| **Message Flow**       | Pull model (consumer fetches)                              | Push model (broker delivers)                          |
| **Scalability**        | High horizontal scaling via partitions                     | Limited; clustering is more complex                   |
| **Throughput**         | Very high (millions of msgs/sec)                           | Moderate (optimized for reliability, not scale)       |
| **Latency**            | Slightly higher (disk-based)                               | Lower (in-memory optimizations)                       |
| **Ordering Guarantee** | Per partition                                              | Strong per queue                                      |
| **Consumer Model**     | Consumer groups with offset control                        | Competing consumers, broker-managed                   |
| **Fault Tolerance**    | Built-in replication (leader/follower, ISR)                | Supported but less scalable                           |
| **Data Model**         | Event streaming (like a log/database)                      | Task/message delivery                                 |
| **Multiple Consumers** | Independent consumption (each gets full stream)            | Competing consumers (one message → one consumer)      |
| **Use Case Focus**     | Event-driven systems, streaming, analytics                 | Task queues, job processing, RPC                      |
| **Example Use Cases**  | Log aggregation, real-time analytics, microservices events | Email queue, background jobs, order processing worker |
| **Protocol Support**   | Kafka protocol (custom)                                    | AMQP, MQTT, STOMP, etc.                               |
| **Complexity**         | Higher (needs proper design)                               | Lower (quick to start)                                |
| **Ecosystem**          | Kafka Streams, Connect, Schema Registry                    | Plugins, exchanges, routing                           |
| **Best For**           | Large-scale, distributed data pipelines                    | Simple messaging and task distribution                |

### Install and Configuration

```bash
sudo apt install default-jdk
sudo apt install default-jre
java -version
```

```bash
cd /opt
wget https://dlcdn.apache.org/kafka/4.2.0/kafka_2.13-4.2.0.tgz
tar -xvf kafka_2.13-4.2.0.tgz
cd kafka_2.13-4.2.0
```

```bash
sudo vi config/kraft/server.properties
```

```bash
process.roles=broker,controller
node.id=1

controller.quorum.voters=1@localhost:9093

listeners=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
advertised.listeners=PLAINTEXT://YOUR_SERVER_IP:9092

controller.listener.names=CONTROLLER
log.dirs=/tmp/kraft-logs
```

```bash
cd /opt/kafka_2.13-4.2.0/bin/
./kafka-storage.sh random-uuid
```

> Should see: piXQZUpUTWW-i56kwq0G1g

```bash
cd /opt/kafka_2.13-4.2.0/bin/
./kafka-storage.sh format \
--standalone \
-t piXQZUpUTWW-i56kwq0G1g \
-c ../config/server.properties
```

Or

```bash
cd /opt/kafka_2.13-4.2.0/bin/
./kafka-storage.sh format \
-t piXQZUpUTWW-i56kwq0G1g \
-c ../config/server.properties
```

```bash
cd /opt/kafka_2.13-4.2.0/bin/
./kafka-metadata-quorum.sh \
--bootstrap-controller localhost:9093 \
describe --status
```

```bash
cd /opt/kafka_2.13-4.2.0/bin/
./kafka-metadata-quorum.sh \
--bootstrap-controller localhost:9093 \
describe --replication
```

### Kafka Components

| Component             | Description                | Key Responsibility           | Important Notes                          |
| --------------------- | -------------------------- | ---------------------------- | ---------------------------------------- |
| **Broker**            | Kafka server instance      | Stores and serves data       | Multiple brokers form a cluster          |
| **Topic**             | Logical stream of messages | Organizes data               | Example: `orders`, `logs`                |
| **Partition**         | Subdivision of a topic     | Enables parallelism          | Ordered, append-only                     |
| **Producer**          | Data sender                | Publishes messages to topics | Can choose partition via key             |
| **Consumer**          | Data reader                | Reads messages from topics   | Pull-based model                         |
| **Consumer Group**    | Group of consumers         | Distributes load             | One partition → one consumer per group   |
| **Offset**            | Message position           | Tracks consumption           | Managed by consumer                      |
| **Controller**        | Special broker role        | Manages cluster state        | Handles leader election                  |
| **Replication**       | Data duplication           | Fault tolerance              | Leader + followers                       |
| **Zookeeper / KRaft** | Metadata manager           | Cluster coordination         | KRaft replaces Zookeeper in modern Kafka |

### Kafka Architecture

![Kafka Architecture](/img/kafka-architecture.png)

### KRaft

Apache Kafka KRaft (Kafka Raft) is a consensus-based metadata management system that uses the Raft algorithm to handle Kafka cluster coordination internally, replacing Apache ZooKeeper.

**Function of KRaft**

Instead of an external coordination system, Kafka itself:

- Elects a controller leader
- Replicates metadata logs
- Maintains cluster agreement using Raft

#### Quorum - Raft Consensus

In an etcd cluster, a `quorum (majority)` of nodes must agree for writes to succeed. The formula is `Quorum = N/2 + 1`. Where N is the total number of etcd nodes.

| Instance (`Node`) | Quorum (`Majority`) | Fault Tolerance (`C1-C2`) |
| :---------------: | :-----------------: | :-----------------------: |
|         1         |          1          |             0             |
|         2         |          2          |             0             |
|       **3**       |        **2**        |           **1**           |
|         4         |          3          |             1             |
|       **5**       |        **3**        |           **2**           |
|         6         |          4          |             2             |
|       **7**       |        **4**        |           **3**           |
|         8         |          5          |             3             |
|         9         |          5          |             4             |

> Where

- Odd number quorum (Min No. of Node) member is Recommended.
- Odd number of Instance/Node/Manager is Recommended.

#### Log Replication

- Every metadata change is written as a log entry
- Examples:
  - topic creation
  - partition changes
  - broker registration
- Leader replicates these logs to followers
- Followers copy and persist them

#### Metadata Management in KRaft

- Metadata is stored inside Kafka itself (not external system)
- Maintained in a replicated log
- Controlled by controller quorum

**What metadata includes:**
- Topics
- Partitions
- ISR (In-Sync Replicas)
- Broker membership
- Configurations

#### Fault Tolerance Mechanism

KRaft is designed for failure handling:

- If a broker/controller fails, system continues running
- A new leader is elected automatically
- As long as majority (quorum) is alive, cluster stays operational

> No single point of failure like ZooKeeper

#### Broker & Controller Roles

**Brokers:**
- Handle producers & consumers
- Store partition data
- Also participate in metadata replication (light role)

**Controllers:**
- Manage cluster metadata
- Run Raft quorum
- Elect leaders for partitions

#### Topic & Partition Handling

When you create a topic:

- Request goes to controller leader
- Metadata is logged
- Log is replicated to quorum
- All brokers update state
- Topic becomes available

### KRaft Mode vs ZooKeeper Mode

| Feature          | ZooKeeper Mode     | KRaft Mode         |
| ---------------- | ------------------ | ------------------ |
| Metadata storage | External ZooKeeper | Internal Kafka log |
| Coordination     | ZooKeeper ensemble | Raft quorum        |
| Complexity       | High               | Lower              |
| Failure handling | Extra dependency   | Built-in           |
| Scalability      | Limited            | Better             |
