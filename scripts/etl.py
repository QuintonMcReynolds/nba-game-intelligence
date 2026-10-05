"""Run the Spark ETL locally over data/lake and write the quality report."""

import json
import shutil

from nbagi import ARTIFACTS, DATA, LAKE
from nbagi.etl_spark import run, spark_session

if __name__ == "__main__":
    tmp = DATA / "spark_tmp"
    spark = spark_session(str(tmp))
    spark.sparkContext.setLogLevel("ERROR")
    report = run(spark, str(LAKE), str(LAKE / "curated"))
    spark.stop()
    shutil.rmtree(tmp, ignore_errors=True)
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "etl_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
