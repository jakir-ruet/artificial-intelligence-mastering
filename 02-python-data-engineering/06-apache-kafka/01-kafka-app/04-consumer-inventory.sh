#!/bin/bash
source ./config.sh

echo "Inventory Service Running..."

$KAFKA_HOME/bin/kafka-console-consumer.sh \
--topic payments \
--group inventory-service \
--from-beginning \
--bootstrap-server $BOOTSTRAP_SERVER
