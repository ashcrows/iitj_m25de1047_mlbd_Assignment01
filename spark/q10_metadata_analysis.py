from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_extract, trim, length, avg, lower,
    regexp_replace, when
)
from pyspark.sql.types import IntegerType
import os


def build_books_df(spark: SparkSession, base_dir: str):
    """
    Loads all Gutenberg .txt files as (file_name, text) DataFrame.
    """
    path = f"file://{os.path.abspath(base_dir)}/*.txt"
    rdd = spark.sparkContext.wholeTextFiles(path)
    df = rdd.toDF(["file_path", "text"])

    # Spark 4.x safe filename extraction
    from pyspark.sql.functions import split, element_at
    df = df.withColumn("file_name", element_at(split("file_path", "/"), -1)) \
           .select("file_name", "text")
    return df


def main():
    spark = SparkSession.builder.appName("Q10_Metadata_Extraction").getOrCreate()

    # Adjust if your dataset folder differs
    base_dir = "q4_wordcount/input/D184MB"

    books_df = build_books_df(spark, base_dir)

    # -----------------------------
    # 2) Regex patterns (Gutenberg header lines)
    # -----------------------------
    # Use (?im):
    #   i = case-insensitive
    #   m = multi-line so ^ matches start-of-line within the big text
    #
    # Patterns capture everything after "Field:" up to end of that line.
    title_re = r"(?im)^\s*Title:\s*(.+)\s*$"
    release_re = r"(?im)^\s*Release Date:\s*(.+)\s*$"
    language_re = r"(?im)^\s*Language:\s*(.+)\s*$"
    encoding_re = r"(?im)^\s*(Character set encoding|Character Set Encoding):\s*(.+)\s*$"

    # Extract fields
    meta_df = (
        books_df
        .withColumn("title", trim(regexp_extract(col("text"), title_re, 1)))
        .withColumn("release_date", trim(regexp_extract(col("text"), release_re, 1)))
        .withColumn("language", trim(regexp_extract(col("text"), language_re, 1)))
        # encoding has 2 groups: group(1) is label, group(2) is the value
        .withColumn("encoding", trim(regexp_extract(col("text"), encoding_re, 2)))
    )

    # Convert empty-string to NULL for cleaner aggregations
    meta_df = (
        meta_df
        .withColumn("title", when(col("title") == "", None).otherwise(col("title")))
        .withColumn("release_date", when(col("release_date") == "", None).otherwise(col("release_date")))
        .withColumn("language", when(col("language") == "", None).otherwise(col("language")))
        .withColumn("encoding", when(col("encoding") == "", None).otherwise(col("encoding")))
    )

    # Extract year from release_date (common in Gutenberg: "... [EBook #12345]")
    # Find first 4-digit year in the release_date line.
    meta_df = meta_df.withColumn(
        "release_year",
        regexp_extract(col("release_date"), r"(\b(18|19|20)\d{2}\b)", 1).cast(IntegerType())
    )

    # -----------------------------
    # 3) Analysis
    # -----------------------------

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

    # -----------------------------
    # 4) Show results (you can redirect to file if needed)
    # -----------------------------
    print("\n=== Sample extracted metadata (5 rows) ===")
    meta_df.select("file_name", "title", "release_date", "release_year", "language", "encoding") \
           .show(5, truncate=False)

    print("\n=== Books released each year ===")
    books_per_year.show(50, truncate=False)

    print("\n=== Most common language ===")
    most_common_language.show(truncate=False)

    print("\n=== Average title length (characters) ===")
    avg_title_len.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
