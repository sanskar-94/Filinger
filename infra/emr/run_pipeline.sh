#!/usr/bin/env bash
# Submit the Phase 1 Spark job to an existing EMR cluster (or EMR Serverless, see below).
# Usage: CLUSTER_ID=j-XXXX BUCKET=filings-lab-dev ./run_pipeline.sh
set -euo pipefail
: "${CLUSTER_ID:?set CLUSTER_ID}"
: "${BUCKET:?set BUCKET}"
LAKE="s3://${BUCKET}/lake"

python -m build --wheel
aws s3 cp dist/filings_lab-0.1.0-py3-none-any.whl "s3://${BUCKET}/code/"
aws s3 cp infra/emr/bootstrap.sh "s3://${BUCKET}/code/"

aws emr add-steps --cluster-id "${CLUSTER_ID}" --steps "[{
  \"Type\": \"Spark\", \"Name\": \"filings-lab-transform\", \"ActionOnFailure\": \"CONTINUE\",
  \"Args\": [\"--deploy-mode\", \"cluster\",
             \"--conf\", \"spark.sql.adaptive.enabled=true\",
             \"--conf\", \"spark.dynamicAllocation.enabled=true\",
             \"--py-files\", \"s3://${BUCKET}/code/filings_lab-0.1.0-py3-none-any.whl\",
             \"s3://${BUCKET}/code/spark_entry.py\",
             \"--index\", \"${LAKE}/bronze/index.parquet\", \"--raw\", \"${LAKE}/bronze/raw\",
             \"--prices\", \"${LAKE}/bronze/prices.parquet\", \"--lake\", \"${LAKE}\"]
}]"

# EMR Serverless alternative:
# aws emr-serverless start-job-run --application-id "$APP_ID" --execution-role-arn "$ROLE_ARN" \
#   --job-driver "{\"sparkSubmit\": {\"entryPoint\": \"s3://${BUCKET}/code/spark_entry.py\",
#     \"entryPointArguments\": [\"--index\",\"${LAKE}/bronze/index.parquet\",\"--raw\",\"${LAKE}/bronze/raw\",
#     \"--prices\",\"${LAKE}/bronze/prices.parquet\",\"--lake\",\"${LAKE}\"],
#     \"sparkSubmitParameters\": \"--conf spark.submit.pyFiles=s3://${BUCKET}/code/filings_lab-0.1.0-py3-none-any.whl\"}}"
