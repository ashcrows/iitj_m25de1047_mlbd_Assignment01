from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_extract, trim, length, avg, lower,
    regexp_replace, when, input_file_name
)
from pyspark.sql.functions import nullif

from pyspark.sql.types import IntegerType
from pyspark.storagelevel import StorageLevel
import os


def build_books_df(spark: SparkSession, base_dir: str):
    """
    Loads all Gutenberg .txt files as (file_name, text) DataFrame.
    Uses Spark reader to get filename robustly.
    """
    path = f"file://{os.path.abspath(base_dir)}/*.txt"

    # Read whole files as text (one row per line), then aggregate per file is painful.
    # wholeTextFiles is correct for "one file => one string".
    rdd = spark.sparkContext.wholeTextFiles(path)
    df = rdd.toDF(["file_path", "text"])

    # Cross-platform: capture last segment after / or \
    df = df.withColumn(
        "file_name",
        regexp_extract(col("file_path"), r"([^/\\]+)$", 1)
    ).select("file_name", "text")

    return df


def main():
    spark = (
        SparkSession.builder
        .appName("Q10_Metadata_Extraction")
        .getOrCreate()
    )

    base_dir = "q4_wordcount/input/D184MB"
    books_df = build_books_df(spark, base_dir)

    # Clean up text lightly: remove BOM, normalize line endings if needed
    books_df = books_df.withColumn("text", regexp_replace(col("text"), r"^\ufeff", ""))

    # Regex patterns (multiline + case-insensitive)
    title_re = r"(?im)^\s*Title:\s*(.+?)\s*$"
    release_re = r"(?im)^\s*Release Date:\s*(.+?)\s*$"
    language_re = r"(?im)^\s*Language:\s*(.+?)\s*$"
    encoding_re = r"(?im)^\s*(Character set encoding|Character Set Encoding):\s*(.+?)\s*$"

    meta_df = (
        books_df
        .withColumn("title", trim(regexp_extract(col("text"), title_re, 1)))
        .withColumn("release_date_raw", trim(regexp_extract(col("text"), release_re, 1)))
        .withColumn("language", trim(regexp_extract(col("text"), language_re, 1)))
        .withColumn("encoding", trim(regexp_extract(col("text"), encoding_re, 2)))
    )

    # Optional cleanup: remove [EBook #...] suffix from release date for cleanliness
    meta_df = meta_df.withColumn(
        "release_date",
        trim(regexp_replace(col("release_date_raw"), r"\s*\[EBook\s*#\d+\]\s*$", ""))
    ).drop("release_date_raw")

    # Convert empty-string to NULL
    for c in ["title", "release_date", "language", "encoding"]:
        meta_df = meta_df.withColumn(c, when(trim(col(c)) == "", None).otherwise(col(c)))

    # Extract first 4-digit year appearing in release_date
    meta_df = meta_df.withColumn(
    "release_year",
    nullif(regexp_extract(col("release_date"), r"(\b(18|19|20)\d{2}\b)", 1), "").cast(IntegerType())
    )

    # Persist since we do multiple actions downstream
    meta_df = meta_df.persist(StorageLevel.MEMORY_AND_DISK)

    # A) number of books released each year
    books_per_year = (
        meta_df
        .where(col("release_year").isNotNull())
        .groupBy("release_year")
        .count()
        .orderBy(col("release_year").asc())
    )

    # B) most common language
    most_common_language = (
        meta_df
        .where(col("language").isNotNull())
        .withColumn("language_norm", trim(lower(col("language"))))
        .groupBy("language_norm")
        .count()
        .orderBy(col("count").desc(), col("language_norm").asc())
        .limit(1)
    )

    # C) average length of titles in characters
    avg_title_len = (
        meta_df
        .where(col("title").isNotNull())
        .select(avg(length(col("title"))).alias("avg_title_length"))
    )

    print("\n=== Sample extracted metadata (5 rows) ===")
    meta_df.select("file_name", "title", "release_date", "release_year", "language", "encoding") \
           .show(5, truncate=False)

    print("\n=== Books released each year ===")
    books_per_year.show(50, truncate=False)

    print("\n=== Most common language ===")
    most_common_language.show(truncate=False)

    print("\n=== Average title length (characters) ===")
    avg_title_len.show(truncate=False)

    # Cleanup
    meta_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
