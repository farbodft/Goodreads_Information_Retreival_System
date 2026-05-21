# 📚 MIR-2026: Goodreads Information Retrieval System

![Goodreads Logo](https://miro.medium.com/v2/resize:fit:1400/1*1An64QITH2du6dsmZQZQxg.png)

### Sharif University of Technology
**Course:** Modern Information Retrieval (Spring 2026)  
**Instructor:** Dr. Mahdieh Soleymani Baghshah

**Student:** Farbod Fattahi

**Student Number:** 402106231

---

## 🔍 Overview

Finding your next great read is more than just looking at a star rating. It requires sophisticated retrieval methods that can handle nuanced summaries and user queries.

This project implements a complete **Information Retrieval (IR) Pipeline** for a Goodreads dataset. From handling near-duplicate summaries to generating contextual snippets and evaluating search relevance, this repository serves as a robust engine for book discovery.

---

## 🚀 Getting Started

### 1. The Dataset
The core data is provided in the repository. You do not need to crawl external sites, but the raw data requires significant preprocessing.
* **Path:** `release/top_3000_rated_books.rar`
* **Note:** Extract this file to your data directory before running the preprocessing scripts.

### 2. Installation
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Running the UI
We use **Streamlit** to provide an interactive search experience.
```bash
streamlit run UI/main.py
```
Once running, navigate to `http://localhost:8501/` in your browser.

---

## 🏗 System Architecture

The project is split into two core modules: **UI** (Interaction) and **Logic** (Processing).

### 🖥 UI Module
Responsible for the search interface and displaying results. It utilizes the `Logic` module to clean queries, correct spells, retrieve the most relevant books, extract snippets, and show the results in a clean dashboard.

### ⚙️ Logic Module
This is the "brain" of the system, comprising 8 critical components:

| # | Component | Description |
| :--- | :--- | :--- |
| 1 | **Preprocess** | Cleans raw text by removing links, punctuation, and stopwords, followed by stemming/lemmatization. |
| 2 | **LSH (Near-Duplicate)** | Uses **MinHashLSH** to identify and remove near-identical book summaries using shingling and LSH. |
| 3 | **Indexing** | Supports multiple inverted indexes (Descriptions, Genres, Characters) and Tiered Indexing. |
| 4 | **Spell Correction** | Fixes query typos using a hybrid Jaccard + Normalized TF score based on the corpus vocabulary. |
| 5 | **Scorer** | Computes relevance scores using **Vector Space Model (VSM)**, **Okapi BM25**, and **Unigram Model**. |
| 6 | **Search Engine** | Orchestrates retrieval via safe/unsafe ranking, merging scores from different document fields. |
| 7 | **Snippet Generation** | Extracts the most relevant "windows" from text, highlighting query tokens with `***` markers. |
| 8 | **Evaluation** | Benchmarks system performance using **MAP, NDCG, MRR, Precision,** and **Recall**. |

---

## 🛠 Deep Dive: Core Logic

### 1️⃣ Text Preprocessing
The `Preprocessor` class converts raw documents into a searchable format:
* **Link Removal**: Strip URLs and emails using RegEx.
* **Normalization**: Case-folding and punctuation removal.
* **Tokenization & Stopwords**: Filter out non-informative words using `stopwords.txt`.
* **Stemming/Lemmatization**: Apply the Stemmer or Lemmatizer to reduce words to their root forms (e.g., "running" → "run").

### 2️⃣ Near-Duplicate Detection (MinHashLSH)
To maintain a high-quality corpus, we detect duplicates using:
* **Shingling**: Convert text into sets of $k$-word shingles (default $k=2$).
* **Min-Hash Signatures**: Compress large sets into compact signatures using permutation-based hashing.
* **LSH Bucketing**: Partition signatures into $b$ bands of $r$ rows. Documents sharing a bucket in any band are considered candidate duplicates.
> **Testing Tip:** Use `LSHFakeData.json` to verify your implementation. Every consecutive pair should be flagged as a duplicate.

### 3️⃣ Advanced Indexing
The system maintains several inverted indexes for different fields:
* **Field Indexes**: Separate indexes for `description`, `genres`, and `characters`.
* **Index Structure**: `{term: {doc_id: tf}}` where `tf` is the frequency of the term in that document.
* **Tiered Indexing**: To optimize search speed, documents are partitioned into levels (Tiers) based on their importance/frequency, allowing "Unsafe Ranking" to only check the top tiers first.

### 4️⃣ Spell Correction Logic
Correcting `whle` to `while` involves two steps:
1. **Candidate Selection**: Find words in the corpus with high **Jaccard Similarity** (k-gram based).
2. **Scoring**: Re-rank the top 5 candidates to ensure we prefer common words over rare ones even if the Jaccard similarity is slightly lower.

### 5️⃣ Scoring Models
You will implement three industry-standard scoring techniques:
* **Vector Space Model (VSM)**: Support for various weighting schemes like `lnc.ltc` (Log-tf, no-idf, cosine-norm for docs; Log-tf, idf, cosine-norm for queries).
* **Okapi BM25**: A non-linear ranking function that improves upon TF-IDF by saturating term frequency and accounting for document length.
* **Unigram Language Model**: Probability-based scoring with three smoothing options:
    * **Naive**: Simple ratio ($tf / dl$).
    * **Bayes**: Dirichlet-style smoothing using collection statistics.
    * **Mixture**: Linear interpolation between document and collection probabilities.

### 6️⃣ Smart Snippets
This class generates a high-quality summary by identifying the optimal windows for each non-stopword in the query. An "optimal window" is defined as the segment containing the highest concentration of query terms. These segments are merged using ellipses (...) to create a concise yet comprehensive snippet. Query words must be highlighted using "***" before and after the word in the final output, and any words from the query absent from the document should be returned separately.

Note: Search and comparison must be performed on a normalized version of the document, but the final snippet must be extracted from the original raw text to maintain readability.

---

## 📈 Evaluation Metrics

The system's effectiveness is measured against a set of queries with known relevant results:
* **MAP (Mean Average Precision)**: Quantifies the quality of the ranked list across multiple queries.
* **NDCG (Normalized Discounted Cumulative Gain)**: Rewards the system for placing highly relevant books at the top of the results.
* **MRR (Mean Reciprocal Rank)**: Measures how long it takes for the user to find the first relevant item.
* And other techniques you learned in the course.

---

**Happy Coding!** 🚀  
Please refer to the docstrings within each `.py` file for implementation-specific signatures and TODO requirements. It is also acceptable to not use the default functions. Creating new ones or deleting existing ones is fine, **as long as the overall architecture of the whole project remains the same.**
