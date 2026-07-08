#!/bin/bash
set -e

# === Apache Airflow Installation Commands ===
AIRFLOW_VERSION=3.0.6
PYTHON_VERSION=3.12

# Update the packages using sudo apt update
sudo apt update

# Install the required Python version along with necessary packages
sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip build-essential

# Create a virtual environment for Airflow and activate it
mkdir -p ~/venv
python${PYTHON_VERSION} -m venv ~/venv/airflow_${AIRFLOW_VERSION}
source ~/venv/airflow_${AIRFLOW_VERSION}/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Define the constraint-file URL
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Install Apache Airflow with the specified version and constraints
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "$CONSTRAINT_URL"

echo "Airflow ${AIRFLOW_VERSION} installed successfully using Python ${PYTHON_VERSION}"
echo "To activate the environment: source ~/venv/airflow_${AIRFLOW_VERSION}/bin/activate"
