# spark/conf/utils.py
import os
from pyspark.sql import functions as F

def build_books_df(spark, input_dir: str):
    """
    Stable (macOS-friendly) loader:
      - reads each file as ONE row using text datasource wholetext=true
    Returns DataFrame:
      - file_path (string)
      - file_name (string)
      - text      (string)
    """

    # Make local path explicit
    if not (input_dir.startswith("hdfs://") or input_dir.startswith("s3://") or input_dir.startswith("file://")):
        input_dir = "file://" + os.path.abspath(input_dir)

    # If user passed a directory, ensure it reads all txt files
    # (safe even if input_dir already has wildcard)
    if input_dir.endswith("/"):
        input_dir = input_dir[:-1]
    if not any(ch in input_dir for ch in ["*", "?", "["]):  # no glob
        input_dir = input_dir + "/*.txt"

    df = (
        spark.read
        .option("wholetext", "true")
        .text(input_dir)  # column: value
        .withColumn("file_path", F.input_file_name())
        .withColumnRenamed("value", "text")
        .withColumn("file_name", F.regexp_extract(F.col("file_path"), r"([^/\\]+)$", 1))
        .select("file_path", "file_name", "text")
    )

    return df
