# Reddit News Digest

Reddit News Digest is an interactive Streamlit application that analyzes Reddit-style discussion datasets and turns topic-specific records into readable, multi-perspective digests.

The project helps users explore what Reddit discussions are saying about a selected topic by filtering relevant records, clustering similar content into perspectives, scoring credibility signals, and presenting the results through a clean dashboard with summaries and visualizations.

---

## What the Project Does

The application allows users to:

- select a Reddit-style CSV dataset
- enter a topic or keyword group
- filter records that strongly match the selected topic
- group similar records into discussion perspectives
- generate readable summaries for each perspective
- calculate transparent credibility signals
- view representative records from each cluster
- explore charts for credibility, sources, subreddits, and clusters
- download the analyzed results as a CSV file

The dashboard is designed to reduce information overload by turning raw Reddit records into a structured digest.

---

## Key Features

- Streamlit-based interactive dashboard
- Automatic CSV column detection
- Support for multiple Reddit-style datasets
- Topic keyword extraction
- Keyword expansion using an alias map
- Exact word matching to reduce false matches
- Weighted relevance scoring
- Minimum relevance threshold to reject weak results
- TF-IDF vectorization for text analysis
- KMeans clustering for grouping similar records
- Cluster keyword extraction
- Heuristic credibility scoring
- Extractive summary generation
- Representative records for each perspective
- Donut chart visualizations
- Data preview and downloadable output

---

## Technologies Used

### Programming Language

- Python

### Web App / Dashboard

- Streamlit

### Data Processing

- pandas
- numpy
- regular expressions
- CSV parsing

### Text Analysis

- TF-IDF vectorization
- English stopword removal
- unigram and bigram feature extraction
- keyword expansion
- exact word matching
- weighted relevance scoring

### Machine Learning

- KMeans clustering from scikit-learn

### Visualization

- matplotlib
- Streamlit metrics
- Streamlit data tables

### Output Formats

- JSON digest output
- CSV analyzed dataset output
- TXT digest output

---

## System Pipeline

```text
Raw Reddit CSV
→ adaptive column detection
→ text cleaning
→ keyword extraction
→ keyword expansion
→ weighted relevance filtering
→ TF-IDF vectorization
→ KMeans clustering
→ credibility scoring
→ cluster summarization
→ Streamlit dashboard
```

---

## How the App Works

### 1. Dataset Selection

The app scans the project folder for CSV files and lets the user choose a dataset from the sidebar.

It supports Reddit-style datasets with columns such as:

```text
title
body
selftext
text
subreddit
score
num_comments
url
created_utc
```

The app automatically detects available columns and maps them to the internal fields used by the pipeline.

---

### 2. Text Preparation

The app creates an `analysis_text` field from the best available text columns.

For post-level datasets:

```text
analysis_text = title + body
```

For comment-level datasets, it uses the best available text field, such as `body` or `text`.

The cleaning stage removes or filters:

- HTML fragments
- extra whitespace
- deleted or removed content
- AutoModerator content
- megathreads or daily discussion threads
- very short records

---

### 3. Keyword and Relevance Filtering

The user enters a topic or keyword group, for example:

```text
climate, weather, storm, flooding, heat
```

The app extracts keywords and expands known terms.

Example:

```text
AI → AI, artificial intelligence, machine learning, automation, ChatGPT, OpenAI
```

The app uses exact word matching so short keywords do not accidentally match unrelated words.

Each record receives a weighted relevance score:

```text
title match      +4.0
body match       +2.0
subreddit match  +1.0
domain match     +0.5
```

This helps prioritize records where the topic appears in important fields like the title.

If too few records match, the app stops and asks the user to try broader or clearer keywords. This prevents the dashboard from generating misleading digests from weak matches.

---

### 4. TF-IDF Text Vectorization

After filtering, the app converts text into numerical vectors using TF-IDF.

TF-IDF gives higher weight to words or phrases that are important within a record but not common across all records.

The vectorizer uses:

```python
TfidfVectorizer(
    stop_words="english",
    max_features=4000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)
```

This means the app uses:

- English stopword removal
- up to 4000 text features
- unigrams and bigrams
- filtering of extremely common terms

---

### 5. KMeans Clustering

The app uses KMeans clustering to group similar Reddit records into perspectives.

The user selects the number of perspectives from the sidebar.

```python
KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)
```

Each cluster becomes one perspective in the final digest.

---

### 6. Cluster Keywords

For each cluster, the app extracts the strongest TF-IDF terms. These terms are shown as keyword chips in the dashboard and help explain what each perspective is mainly about.

---

### 7. Credibility Scoring

The project uses a transparent heuristic credibility score.

The score is not a factual truth label. It is a signal based on available metadata and text characteristics.

The score uses:

- Reddit score or upvotes
- number of comments
- title and body length
- source or domain reliability
- toxicity and misinformation keyword indicators

Engagement values are log-normalized using:

```python
log1p(score)
log1p(num_comments)
```

Credibility levels:

```text
75+     High
55–74   Moderate
35–54   Low
<35     Very Low
```

The credibility score should be interpreted as a helpful signal, not as verified fact-checking.

---

### 8. Summary Generation

For each cluster, the app generates an extractive, template-based summary using:

- top cluster keywords
- strongest representative records
- most common subreddits
- most common linked domains

This approach keeps summaries readable and transparent.

---

## Dashboard Pages

### About

Explains the purpose of the project and gives examples of effective searches.

### Run Analysis

Allows users to configure:

- dataset
- topic keywords
- number of perspectives
- maximum matched records
- minimum relevant records required

### Explore Results

Shows:

- executive summary
- perspective cards
- credibility metrics
- representative records
- visualizations
- data preview
- CSV download option

---

## Visualizations

The app includes charts for:

- records by perspective
- credibility breakdown
- top subreddits
- top linked domains
- high credibility records by cluster
- maximum-score records by cluster

It also includes a perspective snapshot table with:

- perspective number
- cluster ID
- number of records
- percentage share
- credibility score
- credibility level
- top keywords

---

## Datasets

### Included Dataset

This repository includes:

```text
reddit_news.csv
```

This is a smaller Reddit news/post-level dataset used for the main demo.

### Large Dataset

The app can also run on the larger Reddit comments dataset:

```text
kaggle_RC_2019-05.csv
```

However, this file is not included in the GitHub repository because it is approximately 177 MB and exceeds GitHub's standard 100 MB file size limit.

To use the large dataset:

1. Download the Kaggle Reddit Comments May 2019 dataset.
2. Place the CSV file in the project root folder.
3. Make sure it is named:

```text
kaggle_RC_2019-05.csv
```

Expected local structure:

```text
reddit-news-digest/
├── app.py
├── reddit_news.csv
├── kaggle_RC_2019-05.csv
├── requirements.txt
├── README.md
└── output/
```

The app will automatically detect the CSV and show it in the dataset dropdown.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/dipalthaker/reddit-news-digest.git
cd reddit-news-digest
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

On macOS or Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the App

```bash
streamlit run app.py
```

The app will open in your browser.

---

## Recommended Search Examples

Use multiple specific keywords instead of a single broad word.

Good examples:

```text
AI, artificial intelligence, automation, jobs, technology
climate, weather, storm, flooding, heat
election, voting, candidate, government, policy
graduate school, degree, university, students, tuition, education
jobs, layoffs, salary, workers, unemployment, paycheck
```

Avoid vague one-word searches such as:

```text
AI
jobs
politics
masters
news
```

Specific keyword groups produce more relevant and readable digests.

---

## Repository Structure

```text
reddit-news-digest/
├── app.py
├── reddit_news.csv
├── requirements.txt
├── README.md
├── output/
│   └── .gitkeep
└── .gitignore
```

Optional local-only dataset:

```text
kaggle_RC_2019-05.csv
```

This large file is not included in the repository.

---

## Limitations

- The credibility score is heuristic and should not be treated as a verified truth score.
- Reddit data can be noisy, biased, or incomplete.
- Some topics may not have enough matching records in the dataset.
- TF-IDF captures word-based similarity but not deep semantic meaning.
- KMeans requires the number of perspectives to be selected manually.
- The app does not currently perform claim-level fact verification.
- The app does not currently use live Reddit API data.

---

## Future Improvements

Possible future upgrades include:

- claim extraction and factual verification
- Google Fact Check Tools API integration
- ClaimBuster integration
- stronger source reliability database
- semantic embeddings with SentenceTransformers
- UMAP or PCA-based cluster visualization
- real-time Reddit API support
- user personalization based on topic preferences
- adjustable credibility thresholds
- timeline analysis of discussion changes over time

---

## Team

Group 3  
Human-Centered Data Science  
Spring 2026

- Dipal Thaker
- Malhar Gudekar
- Yashvi Bhatt

---

## Disclaimer

This dashboard summarizes Reddit discussion patterns and provides heuristic credibility signals. It should not be interpreted as a factual news verification system. Users should verify important claims through trusted sources, official reports, or professional fact-checking organizations.
