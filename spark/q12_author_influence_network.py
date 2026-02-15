# spark/q12_author_influence_network.py
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from conf.utils import build_books_df


def extract_metadata(df):
    """
    Extract author and release_year from Project Gutenberg-style metadata.
    Returns df with columns: file_name, author, release_year
    """

    # Author line usually like: "Author: Mark Twain"
    author_raw = F.trim(F.regexp_extract(F.col("text"), r"(?im)^\s*Author:\s*(.+?)\s*$", 1))

    # Release Date line usually like: "Release Date: June 1, 2001 [EBook #123]"
    release_date_raw = F.trim(F.regexp_extract(F.col("text"), r"(?im)^\s*Release Date:\s*(.+?)\s*$", 1))

    # Clean "[EBook #...]" suffix if present
    release_date_clean = F.trim(F.regexp_replace(release_date_raw, r"\s*\[EBook\s*#\d+\]\s*$", ""))

    # Extract a 4-digit year from release_date
    year_str = F.regexp_extract(release_date_clean, r"(\b(18|19|20)\d{2}\b)", 1)

    df2 = (
        df.select("file_name", "text")
        .withColumn("author", F.when(author_raw == "", F.lit(None)).otherwise(author_raw))
        .withColumn("release_date", F.when(release_date_clean == "", F.lit(None)).otherwise(release_date_clean))
        .withColumn("release_year", F.when(year_str == "", F.lit(None)).otherwise(year_str).cast(T.IntegerType()))
        .select("file_name", "author", "release_year")
    )

    return df2


def main():
    # Args:
    #   1) input_dir
    #   2) X years window (optional, default 5)
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "q4_wordcount/input/D184MB"
    x_years = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    spark = (
        SparkSession.builder
        .appName("Q12_Author_Influence_Network")
        .config("spark.python.use.daemon", "false")           # reduces Mac BrokenPipe noise
        .config("spark.ui.showConsoleProgress", "false")      # cleaner output
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Step 1: Load books
    books_df = build_books_df(spark, input_dir)

    # Step 2: Extract author + release year
    meta_df = extract_metadata(books_df)

    # Keep only valid rows
    meta_df = meta_df.filter(F.col("author").isNotNull() & F.col("release_year").isNotNull())

    # Step 3: One year per author (earliest year)
    author_year_df = (
        meta_df.groupBy("author")
        .agg(F.min("release_year").alias("author_year"))
        .cache()
    )

    # Step 4: Build edges (author1 -> author2) if within X years and author1 earlier/equal
    a = author_year_df.alias("a")
    b = author_year_df.alias("b")

    edges_df = (
        a.join(
            b,
            on=(
                (F.col("a.author") != F.col("b.author")) &
                (F.col("b.author_year") >= F.col("a.author_year")) &
                ((F.col("b.author_year") - F.col("a.author_year")) <= F.lit(x_years))
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
        .dropDuplicates(["src_author", "dst_author"])
        .cache()
    )

    # Step 5: in-degree / out-degree
    out_deg = (
        edges_df.groupBy("src_author")
        .agg(F.countDistinct("dst_author").alias("out_degree"))
        .withColumnRenamed("src_author", "author")
    )

    in_deg = (
        edges_df.groupBy("dst_author")
        .agg(F.countDistinct("src_author").alias("in_degree"))
        .withColumnRenamed("dst_author", "author")
    )

    degree_df = (
        author_year_df
        .join(in_deg, on="author", how="left")
        .join(out_deg, on="author", how="left")
        .na.fill({"in_degree": 0, "out_degree": 0})
        .orderBy(F.col("author").asc())
        .cache()
    )

    # Step 6: Top 5
    print("\n========== Q12: Author Influence Network ==========")
    print(f"Input dir: {input_dir}")
    print(f"Influence window X (years): {x_years}")
    print(f"Authors (unique): {author_year_df.count()}")
    print(f"Edges (src->dst): {edges_df.count()}")

    print("\n--- Top 5 authors by IN-DEGREE (most influenced) ---")
    degree_df.orderBy(F.col("in_degree").desc(), F.col("author_year").asc()).show(5, truncate=False)

    print("\n--- Top 5 authors by OUT-DEGREE (most influential) ---")
    degree_df.orderBy(F.col("out_degree").desc(), F.col("author_year").asc()).show(5, truncate=False)

    # Show a few edges to prove correctness
    print("\n--- Sample edges (src -> dst) ---")
    edges_df.orderBy(F.col("year_gap").asc(), F.col("src_author").asc()).show(10, truncate=False)

    # Step 7: Theory answers
    print("\n========== Q12: Theory (Short Answers) ==========")

    print("\n1) Representation choice (DataFrame edges)")
    print(
        "Network is represented as a DataFrame of edges (src_author, dst_author).\n"
        "Advantages: SQL-like joins/aggregations, Catalyst optimization, easy degree calculations.\n"
        "Disadvantages: Not a full graph library (no PageRank/communities unless using GraphFrames/GraphX)."
    )

    print("\n2) Effect of time window X")
    print(
        "Smaller X => fewer edges, sparser graph, fewer potential influences.\n"
        "Larger X => more edges, denser graph, more connections.\n"
        "Limitation: time proximity does not prove influence; true influence depends on genre, citations, geography, popularity, etc."
    )

    print("\n3) Scalability + Spark optimizations")
    print(
        "Main cost is the author self-join. With millions of authors, naive pairwise is too expensive.\n"
        "Optimizations:\n"
        "- Bucket authors by year (or year-range) to reduce join candidates.\n"
        "- Partition by year and do range-join within partitions.\n"
        "- Use approximate/graph methods if needed.\n"
        "- Cache reused DataFrames and select only required columns."
    )

    spark.stop()


if __name__ == "__main__":
    main()
