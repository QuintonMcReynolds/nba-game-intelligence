#!/usr/bin/env bash
# Deploy the lake + Glue ETL + Athena stack and run the ETL once.
# NOT run as part of this repo's build. Creates billable AWS resources (S3 storage, Glue DPU
# hours: ~4 workers x a few minutes per run, Athena per TB scanned).
#   usage: AWS_PROFILE=... ./cloud/deploy.sh [project-name]
set -euo pipefail
PROJECT="${1:-nba-game-intelligence}"
STACK="$PROJECT"
cd "$(dirname "$0")/.."

aws cloudformation deploy --stack-name "$STACK" --template-file cloud/template.yaml \
  --capabilities CAPABILITY_IAM --parameter-overrides ProjectName="$PROJECT"
BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='Bucket'].OutputValue" --output text)

aws s3 cp src/nbagi/etl_spark.py "s3://$BUCKET/code/etl_spark.py"
python scripts/ingest.py                                  # raw Parquet, locally
aws s3 sync data/lake/raw_pbp "s3://$BUCKET/lake/raw_pbp"
RUN=$(aws glue start-job-run --job-name "$PROJECT-etl" --query JobRunId --output text)
echo "Glue job run $RUN started; the crawler runs when it succeeds."
echo "Then query in Athena (workgroup $PROJECT), e.g. sql/era_trends.sql with the"
echo "read_parquet(...) paths replaced by the catalog tables events/games."
