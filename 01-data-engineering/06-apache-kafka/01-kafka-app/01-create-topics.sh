#!/bin/bash
source ./config.sh

echo "Creating topics..."

$KAFKA_HOME/bin/kafka-topics.sh --create \
--topic orders \
--bootstrap-server $BOOTSTRAP_SERVER \
--partitions 3 \
--replication-factor 1

$KAFKA_HOME/bin/kafka-topics.sh --create \
--topic payments \
--bootstrap-server $BOOTSTRAP_SERVER \
--partitions 3 \
--replication-factor 1

$KAFKA_HOME/bin/kafka-topics.sh --create \
--topic notifications \
--bootstrap-server $BOOTSTRAP_SERVER \
--partitions 3 \
--replication-factor 1

echo "Topics created successfully"
