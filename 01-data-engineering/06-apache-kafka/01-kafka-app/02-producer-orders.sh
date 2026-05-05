#!/bin/bash

KAFKA_HOME=/opt/kafka_2.13-4.2.0
BOOTSTRAP=localhost:9092

echo "Starting Order Producer..."

$KAFKA_HOME/bin/kafka-console-producer.sh \
--topic orders \
--bootstrap-server $BOOTSTRAP
