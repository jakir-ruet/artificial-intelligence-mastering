"""
Airflow 3.0.6 Hands-on Example
A complete ETL pipeline demonstrating core Airflow concepts
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
import json
import random

# Default arguments applied to all tasks in the DAG
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(
    dag_id='etl_tutorial_dag',
    default_args=default_args,
    description='First ETL pipeline: Extract → Transform → Load',
    schedule_interval='@daily',  # Runs once per day
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't run past missed schedules
    tags=['tutorial', 'etl'],
    max_active_runs=1  # Only one active run at a time
)
def etl_tutorial_dag():
    """
    ETL Pipeline Example:
    - Extract: Generate or fetch raw data
    - Transform: Clean and process the data
    - Load: Save the processed data
    """

    @task
    def extract():
        """Step 1: Extract raw data from source"""
        print("📊 Starting extraction...")

        # Simulate extracting data from an API or database
        raw_data = {
            'timestamp': datetime.now().isoformat(),
            'transactions': [
                {'id': 1, 'amount': 100, 'product': 'laptop', 'status': 'completed'},
                {'id': 2, 'amount': 50, 'product': 'mouse', 'status': 'completed'},
                {'id': 3, 'amount': 25, 'product': 'keyboard', 'status': 'pending'},
                {'id': 4, 'amount': 200, 'product': 'monitor', 'status': 'completed'},
                {'id': 5, 'amount': 75, 'product': 'headphones', 'status': 'failed'}
            ]
        }

        print(f"✅ Extracted {len(raw_data['transactions'])} transactions")

        # Save to XCom for next tasks (automatically handled)
        return raw_data

    @task
    def transform(extracted_data):
        """Step 2: Transform and clean the extracted data"""
        print("🔄 Starting transformation...")

        # Filter only completed transactions
        completed = [
            t for t in extracted_data['transactions']
            if t['status'] == 'completed'
        ]

        # Calculate total sales
        total_sales = sum(t['amount'] for t in completed)
        avg_sales = total_sales / len(completed) if completed else 0

        # Transform the data structure
        transformed_data = {
            'processed_at': datetime.now().isoformat(),
            'total_transactions': len(extracted_data['transactions']),
            'completed_transactions': len(completed),
            'total_sales': total_sales,
            'average_sale_amount': round(avg_sales, 2),
            'failed_transactions': len([t for t in extracted_data['transactions'] if t['status'] == 'failed']),
            'products_sold': [t['product'] for t in completed]
        }

        print(f"✅ Transform complete: ${total_sales} total sales from {len(completed)} transactions")
        return transformed_data

    @task
    def load(transformed_data):
        """Step 3: Load the transformed data"""
        print("💾 Starting load process...")

        # In a real scenario, you would save to a database
        # For this example, we'll just create a JSON file
        output_file = f"/tmp/airflow_etl_output_{datetime.now().strftime('%Y%m%d')}.json"

        with open(output_file, 'w') as f:
            json.dump(transformed_data, f, indent=2)

        print(f"✅ Data loaded successfully to {output_file}")

        # Also print a summary
        print("\n" + "="*50)
        print("📈 ETL SUMMARY")
        print("="*50)
        print(f"Total Transactions: {transformed_data['total_transactions']}")
        print(f"Completed: {transformed_data['completed_transactions']}")
        print(f"Failed: {transformed_data['failed_transactions']}")
        print(f"Total Sales: ${transformed_data['total_sales']}")
        print(f"Average Sale: ${transformed_data['average_sale_amount']}")
        print("="*50)

        return f"Data saved to {output_file}"

    @task
    def generate_report(load_result):
        """Bonus: Generate a report"""
        print("\n📋 Generating completion report...")
        print(f"Report: {load_result}")
        print("✅ ETL Pipeline completed successfully!")
        return True

    # Define task dependencies
    extracted = extract()
    transformed = transform(extracted)
    loaded = load(transformed)
    report = generate_report(loaded)

# Instantiate the DAG
dag_instance = etl_tutorial_dag()
