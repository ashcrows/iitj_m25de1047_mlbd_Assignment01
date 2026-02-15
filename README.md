# iitj_m25de1047_mlbd_Assignment01

This repository contains solutions for Assignment 01 of the MLwDE (Machine Learning with Data Engineering) course at IIT Jodhpur.

## Project Overview

The project demonstrates distributed data processing using Hadoop MapReduce and Apache Spark, focusing on text analysis of Project Gutenberg books.

### Main Components

#### 1. Hadoop MapReduce Word Count
- **Location:** `q4_wordcount/src/WordCount.java`
- **Description:** Classic MapReduce job to count word frequencies in large text files. Handles punctuation and case normalization.
- **Usage:** 
  ```
  hadoop jar wordcount.jar WordCount <input_path> <output_path>
  ```
- **Input:** Text files in `q4_wordcount/input/`
- **Output:** Word counts in `q4_wordcount/output_local/` or HDFS output directory.

#### 2. Spark Data Analysis Scripts
- **Location:** `spark/`
- **Scripts:**
  - `q10_metadata_analysis.py`: Extracts and analyzes metadata (title, author, release year, language, encoding) from Gutenberg books.
  - `q11_tfidf_similarity.py`: Computes TF-IDF vectors for each book and finds the most similar books to a target using cosine similarity.
  - `q12_author_influence_network.py`: Builds an author influence network based on book release years and analyzes in/out degree of influence.
  - `load_books_df.py` & `conf/utils.py`: Utility functions for loading and preprocessing book data.

#### 3. Data
- **Sample Inputs:** 
  - `q4_wordcount/input/file1.txt`, `wordcount/input/file1.txt`
  - Large dataset: `q4_wordcount/input/D184MB/` (Project Gutenberg books)
- **Sample Outputs:** 
  - `q4_wordcount/output_q7.txt`, HDFS output files

## Folder Structure

- `q4_wordcount/` – MapReduce code, input/output data, and build artifacts
- `spark/` – PySpark scripts for advanced text and network analysis
- `wordcount/` – Additional word count input samples
- `.gitignore` – Ignores build, data, and zip files as appropriate

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
2. Install dependencies for Hadoop and Spark as per your environment.
3. Place input data in the appropriate folders.
4. Run MapReduce or Spark jobs as described above.

## Notes

- The project is intended for academic use and demonstrates scalable text processing.
- For any issues, contact the course instructor or TA.

---
