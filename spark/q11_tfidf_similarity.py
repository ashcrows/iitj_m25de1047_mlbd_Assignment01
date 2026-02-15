# spark/q12_author_influence_network.py
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from conf.utils import build_books_df


def extract_first_group(text_col, pattern: str, group_idx: int = 1):
    """
    Helper: regexp_extract returns "" when not found.
    Convert empty -> NULL, trim whitespace.
    """
    v = F.trim(F.regexp_extract(text_col, pattern, group_idx))
    return F.when(v == "", F.lit(None)).otherwise(v)


def clean_gutenberg_line(s):
    """
    Remove common trailing artifacts like [EBook #123] and extra whitespace.
    """
    s = F.regexp_replace(s, r"\s*\[EBook\s*#\d+\]\s*$", "")
    s = F.regexp_replace(s, r"\s+", " ")
    return F.trim(s)


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "q4_wordcount/input/D184MB"
    X = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    spark = (
        SparkSession.builder
        .appName("Q12_Author_Influence_Network")
        # macOS Spark stability
        .config("spark.python.use.daemon", "false")
        .config("spark.python.worker.reuse", "false")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    books_df = build_books_df(spark, input_dir)

    # -----------------------------
    # STEP 1: Extract Author + Release Date (robust)
    # -----------------------------
    # Match lines like:
    #   Author: Mark Twain
    #   Authors: A; B
    #   Release Date: January 1, 2000 [EBook #123]
    #
    # (?im) => case-insensitive + multiline
    author_pat = r"(?im)^\s*Author[s]?\s*:\s*(.+?)\s*$"
    release_pat = r"(?im)^\s*Release\s+Date\s*:\s*(.+?)\s*$"

    meta = (
        books_df
        .withColumn("author_raw", extract_first_group(F.col("text"), author_pat, 1))
        .withColumn("release_date_raw", extract_first_group(F.col("text"), release_pat, 1))
        .withColumn("author", clean_gutenberg_line(F.col("author_raw")))
        .withColumn("release_date", clean_gutenberg_line(F.col("release_date_raw")))
    )

    # Extract 4-digit year from release_date (supports 18xx/19xx/20xx anywhere)
    meta = meta.withColumn(
        "release_year",
        F.when(
            F.col("release_date").isNull(),
            F.lit(None).cast(IntegerType())
        ).otherwise(
            F.regexp_extract(F.col("release_date"), r"(\b(18|19|20)\d{2}\b)", 1).cast(IntegerType())
        )
    )

    # Keep only rows where we have author + year
    authors_df = (
        meta
        .select("file_name", "author", "release_year")
        .where(F.col("author").isNotNull() & F.col("release_year").isNotNull())
    )

    # -----------------------------
    # DEBUG: Show why 0 authors happens
    # -----------------------------
    total_files = books_df.count()
    extracted = authors_df.count()

    print("\n========== Q12: Author Influence Network ==========")
    print(f"Input dir: {input_dir}")
    print(f"Influence window X (years): {X}")
    print(f"Total files read: {total_files}")
    print(f"Books with (author + year): {extracted}")

    if extracted == 0:
        print("\n[DEBUG] No books matched author/year. Showing 20 sample extracted raw lines:")
        meta.select("file_name", "author_raw", "release_date_raw").show(20, truncate=False)

        print("\n[DEBUG] Showing 30 lines from one sample file to inspect formatting (first file):")
        one = books_df.select("file_name", "text").limit(1).collect()
        if one:
            sample_text = one[0]["text"]
            # print only first ~60 lines
            print("\n".join(sample_text.splitlines()[:60]))

        spark.stop()
        sys.exit(0)

    # -----------------------------
    # STEP 2: Reduce to one year per author (simplification)
    # -----------------------------
    # If an author has multiple books, choose earliest release_year (author "first appearance")
    author_year_df = (
        authors_df
        .groupBy("author")
        .agg(F.min("release_year").alias("author_year"))
    )

    # -----------------------------
    # STEP 3: Build influence edges (src -> dst)
    # Definition: src potentially influenced dst if:
    #   src_year <= dst_year and (dst_year - src_year) <= X and src != dst
    # -----------------------------
    a = author_year_df.alias("a")
    b = author_year_df.alias("b")

    edges = (
        a.join(
            b,
            on=(
                (F.col("a.author") != F.col("b.author")) &
                (F.col("a.author_year") <= F.col("b.author_year")) &
                (F.col("b.author_year") - F.col("a.author_year") <= F.lit(X))
            ),
            how="inner"
        )
        .select(
            F.col("a.author").alias("src_author"),
            F.col("b.author").alias("dst_author"),
            F.col("a.author_year").alias("src_year"),
            F.col("b.author_year").alias("dst_year"),
            (F.col("b.author_year") - F.col("a.author_year")).alias("year_gap")
        )
    )

    # -----------------------------
    # STEP 4: In-degree and Out-degree
    # -----------------------------
    out_deg = edges.groupBy("src_author").count().withColumnRenamed("count", "out_degree")
    in_deg = edges.groupBy("dst_author").count().withColumnRenamed("count", "in_degree")

    degrees = (
        author_year_df
        .withColumnRenamed("author", "author_key")
        .join(out_deg, author_year_df.author == out_deg.src_author, "left")
        .join(in_deg, author_year_df.author == in_deg.dst_author, "left")
        .select(
            author_year_df.author.alias("author"),
            "author_year",
            F.coalesce(F.col("in_degree"), F.lit(0)).alias("in_degree"),
            F.coalesce(F.col("out_degree"), F.lit(0)).alias("out_degree"),
        )
    )

    print(f"Authors (unique): {author_year_df.count()}")
    print(f"Edges (src->dst): {edges.count()}")

    print("\n--- Top 5 authors by IN-DEGREE (most influenced) ---")
    degrees.orderBy(F.col("in_degree").desc(), F.col("author")).show(5, truncate=False)

    print("\n--- Top 5 authors by OUT-DEGREE (most influential) ---")
    degrees.orderBy(F.col("out_degree").desc(), F.col("author")).show(5, truncate=False)

    print("\n--- Sample edges (src -> dst) ---")
    edges.orderBy(F.col("year_gap").asc(), F.col("src_author"), F.col("dst_author")).show(10, truncate=False)

    # -----------------------------
    # STEP 5: Theory answers (required)
    # -----------------------------
    print("\n========== Q12: Theory (Short Answers) ==========")

    print("\n1) Representation choice (DataFrame edges)")
    print(
        "Network is represented as a DataFrame of edges (src_author, dst_author).\n"
        "Advantages: SQL-like joins/aggregations, Catalyst optimization, easy degree calculations.\n"
        "Disadvantages: Not a full graph library (no PageRank/communities unless GraphFrames/GraphX is used)."
    )

    print("\n2) Effect of time window X")
    print(
        "Smaller X => fewer edges (sparser network).\n"
        "Larger X => more edges (denser network).\n"
        "Limitation: time closeness does not prove true influence; real influence depends on genre, citations, popularity, etc."
    )

    print("\n3) Scalability + Spark optimizations")
    print(
        "Main cost is the author self-join.\n"
        "For very large datasets, optimizations include:\n"
        "- Bucket by year (or year ranges) to reduce join candidates.\n"
        "- Partition/sort by year and use range-join conditions.\n"
        "- Cache reused DataFrames, select only needed columns.\n"
        "- Consider approximate graph methods if needed."
    )

    spark.stop()


if __name__ == "__main__":
    main()
