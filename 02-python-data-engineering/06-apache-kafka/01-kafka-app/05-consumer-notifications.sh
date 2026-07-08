#!/bin/bash
source ./config.sh

echo "Notification Service Running..."

$KAFKA_HOME/bin/kafka-console-consumer.sh \
--topic notifications \
--group notification-service \
--from-beginning \
--bootstrap-server $BOOTSTRAP_SERVER
