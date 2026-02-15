# spark/conf/utils.py

import os
from pyspark.sql import functions as F


def build_books_df(spark, input_dir: str):
    """
    Loads Gutenberg .txt files as (file_path, text, file_name).
    Automatically handles local paths.
    """

    # Convert local path to file:// if needed
    if not (
        input_dir.startswith("hdfs://")
        or input_dir.startswith("s3://")
        or input_dir.startswith("file://")
    ):
        input_dir = "file://" + os.path.abspath(input_dir)

    rdd = spark.sparkContext.wholeTextFiles(input_dir)

    df = rdd.toDF(["file_path", "text"])

    # Extract filename safely (works on Mac/Linux/Windows paths)
    df = df.withColumn(
        "file_name",
        F.regexp_extract(F.col("file_path"), r"([^/\\\\]+)$", 1)
    )

    return df
