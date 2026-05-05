#!/bin/bash
source ./config.sh

echo "Payment Service Running..."

$KAFKA_HOME/bin/kafka-console-consumer.sh \
--topic orders \
--group payment-service \
--from-beginning \
--bootstrap-server $BOOTSTRAP_SERVER
