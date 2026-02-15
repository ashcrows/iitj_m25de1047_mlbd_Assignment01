# spark/q11_tfidf_similarity.py

import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer,
)
from pyspark.ml.linalg import Vector
from conf.utils import build_books_df


def strip_gutenberg_header_footer(text_col):
    """
    Removes Project Gutenberg header and footer.
    """

    start_pat = r"(?is)\*\*\*\s*start of (the )?project gutenberg.*?\*\*\*"
    end_pat = r"(?is)\*\*\*\s*end of (the )?project gutenberg.*?\*\*\*"

    after_start = F.regexp_extract(text_col, start_pat + r"(.*)", 2)
    cleaned = F.when(F.length(after_start) > 0, after_start).otherwise(text_col)

    cleaned = F.regexp_replace(cleaned, end_pat + r".*$", "")

    return cleaned


def cosine_similarity(normalized_vec: Vector, target_vec: Vector) -> float:
    # Since vectors are normalized, cosine similarity = dot product
    return float(normalized_vec.dot(target_vec))


def main():

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "q4_wordcount/input/D184MB"
    target_file = sys.argv[2] if len(sys.argv) > 2 else "10.txt"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    spark = (
        SparkSession.builder
        .appName("Q11_TFIDF_Book_Similarity")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    books_df = build_books_df(spark, input_dir)

    # -------------------------
    # Text Preprocessing
    # -------------------------

    books_df = books_df.withColumn(
        "clean_text",
        strip_gutenberg_header_footer(F.col("text"))
    )

    books_df = books_df.withColumn("clean_text", F.lower(F.col("clean_text")))
    books_df = books_df.withColumn(
        "clean_text",
        F.regexp_replace(F.col("clean_text"), r"[^a-z0-9\s]+", " ")
    )

    # -------------------------
    # TF-IDF Pipeline
    # -------------------------

    tokenizer = RegexTokenizer(
        inputCol="clean_text",
        outputCol="tokens",
        pattern=r"\s+",
        minTokenLength=2,
    )

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered"
    )

    cv = CountVectorizer(
        inputCol="filtered",
        outputCol="tf",
        vocabSize=200000,
        minDF=2,
    )

    idf = IDF(
        inputCol="tf",
        outputCol="tfidf"
    )

    normalizer = Normalizer(
        inputCol="tfidf",
        outputCol="tfidf_norm",
        p=2.0
    )

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, normalizer])
    model = pipeline.fit(books_df)

    feat_df = model.transform(books_df).select("file_name", "tfidf_norm")

    # -------------------------
    # Target Vector
    # -------------------------

    target_row = (
        feat_df
        .filter(F.col("file_name") == target_file)
        .limit(1)
        .collect()
    )

    if not target_row:
        print(f"[ERROR] Target file '{target_file}' not found.")
        feat_df.select("file_name").orderBy("file_name").show(20, truncate=False)
        spark.stop()
        sys.exit(1)

    target_vec = target_row[0]["tfidf_norm"]
    bc_target = spark.sparkContext.broadcast(target_vec)

    sim_udf = F.udf(
        lambda v: cosine_similarity(v, bc_target.value),
        "double"
    )

    result = (
        feat_df
        .withColumn("cosine_similarity", sim_udf(F.col("tfidf_norm")))
        .filter(F.col("file_name") != target_file)
        .orderBy(F.col("cosine_similarity").desc())
        .select("file_name", "cosine_similarity")
        .limit(top_k)
    )

    print("\n========== Q11: Top Similar Books ==========")
    print(f"Target book: {target_file}")
    result.show(top_k, truncate=False)

    # -------------------------
    # Theory Answers
    # -------------------------

    print("\n========== Q11: Theory Answers ==========")

    print("\n1) TF, IDF and Why TF-IDF?")
    print(
        "TF measures how frequently a term appears in a document.\n"
        "IDF measures how rare a term is across all documents.\n"
        "TF-IDF combines both so common words get low weight and meaningful words get higher importance."
    )

    print("\n2) Cosine Similarity and Why Use It?")
    print(
        "Cosine similarity measures the angle between two vectors.\n"
        "It is suitable for TF-IDF because it compares content similarity independent of document length."
    )

    print("\n3) Scalability Challenges and Spark’s Role")
    print(
        "Pairwise similarity is O(N²), which is expensive for large datasets.\n"
        "Spark distributes TF-IDF and similarity computations across nodes.\n"
        "Optimizations include computing similarity only for a query document or using approximate methods like LSH."
    )

    spark.stop()


if __name__ == "__main__":
    main()
