# spark/conf/utils.py
import os
from pyspark.sql import functions as F

def build_books_df(spark, input_dir: str):
    """
    Returns DataFrame:
      - file_path (string)
      - file_name (string)
      - text      (string)
    Reads each file as a single record using wholeTextFiles().
    """

    # If user passed a local path, make it explicit
    if not (input_dir.startswith("hdfs://") or input_dir.startswith("s3://") or input_dir.startswith("file://")):
        input_dir = "file://" + os.path.abspath(input_dir)

    rdd = spark.sparkContext.wholeTextFiles(input_dir)
    df = rdd.toDF(["file_path", "text"])

    # Extract just the filename from path (works for file://, hdfs://, s3://)
    df = df.withColumn("file_name", F.regexp_extract(F.col("file_path"), r"([^/\\]+)$", 1))

    return df
