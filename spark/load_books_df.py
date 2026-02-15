from pyspark.sql import SparkSession
from pyspark.sql.functions import split

# Create Spark session
spark = SparkSession.builder.appName("GutenbergBooksLoader").getOrCreate()

# Local dataset path (your repo path)
path = "q4_wordcount/input/D184MB/*.txt"

# Read each file as (path, full_text)
rdd = spark.sparkContext.wholeTextFiles(path)

# Convert to DataFrame: file_path, text
books_df = rdd.toDF(["file_path", "text"])

# Extract file_name from the path
books_df = books_df.withColumn("file_name", split(books_df.file_path, "/").getItem(-1)) \
                   .select("file_name", "text")

# Show and schema (for submission proof)
books_df.show(5, truncate=False)
books_df.printSchema()

spark.stop()
