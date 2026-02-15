from pyspark.sql import SparkSession
from pyspark.sql.functions import split
import os

spark = SparkSession.builder.appName("GutenbergBooksLoader").getOrCreate()

base_dir = os.path.abspath("q4_wordcount/input/D184MB")
path = f"file://{base_dir}/*.txt"

rdd = spark.sparkContext.wholeTextFiles(path)

books_df = rdd.toDF(["file_path", "text"])
books_df = books_df.withColumn("file_name", split(books_df.file_path, "/").getItem(-1)) \
                   .select("file_name", "text")

books_df.show(5, truncate=False)
books_df.printSchema()

spark.stop()
