# Human-Centered Personalized Information Subscription Service (HCDS)

A comprehensive Python system that aggregates social media posts about current events and generates daily digests summarizing multiple viewpoints with credibility indicators.

## 📋 Project Overview

The HCDS system transforms large volumes of unstructured social media discussions into concise, credible summaries. It identifies distinct perspectives on trending topics and presents them with transparency about source reliability.

### Key Features

- **Multi-Perspective Summarization**: Automatically discovers and summarizes different viewpoints on a topic
- **Credibility Scoring**: Evaluates post trustworthiness based on engagement, linguistic quality, and misinformation signals
- **Thematic Clustering**: Groups semantically similar posts using sentence embeddings
- **Automatic Field Detection**: Adapts to different dataset structures (missing fields handled gracefully)
- **Multiple Output Formats**: Generates both human-readable digests and structured JSON
- **Detailed Analytics**: Provides statistics on perspectives, engagement, and credibility

## 📊 Pipeline Architecture

```
Raw Data
    ↓
[1] Data Preprocessing
    • Remove duplicates, URLs, emojis
    • Clean and normalize text
    • Extract engagement features
    ↓
Cleaned Data
    ↓
[2] Embedding & Clustering
    • Generate sentence embeddings (SentenceTransformers)
    • KMeans clustering into themes
    • Extract representative keywords
    ↓
Thematic Clusters
    ↓
[3] Multi-Perspective Summarization
    • Generate 3-4 sentence summaries per cluster
    • Extract representative example posts
    • Compute engagement statistics
    ↓
Perspective Summaries
    ↓
[4] Credibility Scoring
    • Engagement-based scoring (25%)
    • Linguistic quality (25%)
    • Source reliability (20%)
    • Toxicity detection (15%)
    • Misinformation signals (15%)
    ↓
Scored Perspectives
    ↓
[5] Digest Generation
    • Format multiple perspectives
    • Rank by credibility
    • Create readable digest
    ↓
Daily Digest Output
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see Installation)

### Installation

1. **Clone or extract the project**:
   ```bash
   cd HCDS\ Project
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install pandas numpy scikit-learn sentence-transformers transformers
   ```

### Basic Usage

```python
from src.main import HCDSPipeline

# Configuration
config = {
    'n_clusters': 5,
    'embedding_model': 'all-MiniLM-L6-v2',
    'sample_size': None,  # Set to number to debug with smaller sample
}

# Initialize pipeline
pipeline = HCDSPipeline(config=config)

# Run full pipeline
text_digest, json_digest = pipeline.run_full_pipeline(
    data_path='kaggle_RC_2019-05.csv',
    event_title='Current Events Discussion',
    output_dir='output'
)

# Print digest
print(text_digest)
```

## 📁 Project Structure

```
HCDS Project/
├── src/
│   ├── preprocessing.py         # Data loading and cleaning
│   ├── clustering.py             # Thematic clustering
│   ├── summarization.py          # Multi-perspective summaries
│   ├── credibility.py            # Credibility scoring
│   ├── digest_generator.py       # Output formatting
│   └── main.py                   # Pipeline orchestrator
├── notebooks/
│   └── demo_pipeline.ipynb       # Interactive demonstration
├── data/
│   └── (processed outputs)
├── kaggle_RC_2019-05.csv         # Input dataset
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📘 Module Documentation

### 1. **preprocessing.py** - DataPreprocessor

Handles data loading, cleaning, and feature engineering.

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(min_text_length=10)

# Load data (auto-detects fields)
df = preprocessor.load_data('kaggle_RC_2019-05.csv', sample_size=5000)

# Clean and preprocess
cleaned_df = preprocessor.preprocess(df)

# Get statistics
stats = preprocessor.get_summary_stats(cleaned_df)
```

**Features**:
- Adaptive field mapping (handles missing columns)
- URL/emoji/special character removal
- Text normalization and lowercasing
- Stopword removal
- Length-based filtering
- Engagement metric normalization
- Text quality indicators (caps ratio, punctuation, etc.)

### 2. **clustering.py** - ThematicClusterer

Performs semantic clustering using sentence embeddings.

```python
from src.clustering import ThematicClusterer

clusterer = ThematicClusterer(n_clusters=5)

# Generate embeddings
embeddings = clusterer.generate_embeddings(texts)

# Cluster posts
labels = clusterer.cluster(embeddings)

# Get cluster summary
summary = clusterer.get_cluster_summary(df, embeddings)
```

**Features**:
- SentenceTransformers embeddings
- KMeans clustering
- Optimal cluster detection (elbow method)
- Keyword extraction per cluster
- Cluster statistics

### 3. **summarization.py** - PerspectiveSummarizer

Generates summaries and example posts for each perspective.

```python
from src.summarization import PerspectiveSummarizer

summarizer = PerspectiveSummarizer(summary_length=3)

# Generate abstractive summaries (uses transformers)
perspectives = summarizer.generate_perspective_summaries(
    df, keywords_dict, embeddings
)

# Extract example posts
examples = summarizer.extract_example_posts(cluster_df)
```

**Features**:
- Abstractive summarization (BART)
- Extractive summarization (fallback)
- Example post extraction with quality scoring
- Engagement statistics per perspective

### 4. **credibility.py** - CredibilityScorer

Scores credibility based on multiple signals.

```python
from src.credibility import CredibilityScorer

scorer = CredibilityScorer()

# Score all posts
df = scorer.score_dataframe(df)

# Get cluster-level scores
cluster_scores = scorer.score_clusters(df)

# Get summary statistics
summary = scorer.get_credibility_summary(df)
```

**Features**:
- Multi-factor credibility scoring (0-100 scale)
- Engagement metrics scoring
- Linguistic quality assessment
- Toxicity detection
- Misinformation pattern detection
- Source reliability baseline

**Credibility Formula**:
```
Score = 0.25 * engagement 
      + 0.25 * linguistic_quality
      + 0.20 * source_reliability
      + 0.15 * toxicity_score
      + 0.15 * misinformation_score
```

### 5. **digest_generator.py** - DigestGenerator

Generates formatted output in multiple formats.

```python
from src.digest_generator import DigestGenerator

generator = DigestGenerator(max_perspectives=5)

# Generate text digest
text = generator.generate_text_digest(
    event_title, perspectives, cluster_credibility
)

# Generate JSON digest
json_data = generator.generate_json_digest(
    event_title, perspectives, cluster_credibility
)

# Save outputs
generator.save_digest_text(text, 'output.txt')
generator.save_digest_json(json_data, 'output.json')
```

**Output Formats**:
- Human-readable text format with section headers
- Structured JSON with metadata
- Comparison tables
- Executive summaries

### 6. **main.py** - HCDSPipeline

Orchestrates the complete workflow.

```python
from src.main import HCDSPipeline

pipeline = HCDSPipeline(config={
    'n_clusters': 5,
    'embedding_model': 'all-MiniLM-L6-v2',
    'sample_size': None,
})

text_digest, json_digest = pipeline.run_full_pipeline(
    'kaggle_RC_2019-05.csv',
    'Event Title',
    'output'
)

stats = pipeline.get_statistics()
```

## 🔧 Configuration

Customize pipeline behavior via config dictionary:

```python
config = {
    'n_clusters': 5,                    # Number of topic clusters
    'embedding_model': 'all-MiniLM-L6-v2',  # SentenceTransformer model
    'min_text_length': 10,              # Minimum post length after cleaning
    'summary_length': 3,                # Sentences in generated summary
    'examples_per_cluster': 3,          # Example posts per perspective
    'max_perspectives': 5,              # Perspectives to include in digest
    'sample_size': None,                # None = use all; int = sample
    'auto_detect_clusters': False,      # Auto-detect optimal k using elbow method
}
```

## 📊 Example Output

```
================================================================================
DAILY DIGEST - MULTIPLE PERSPECTIVES
================================================================================

Generated: 2024-01-15
Source: Multiple sources
Total Posts Analyzed: 47832

--------------------------------------------------------------------------------
EVENT/TOPIC: Current Events Discussion - May 2019
--------------------------------------------------------------------------------

================================================================================
PERSPECTIVE 1
================================================================================

Posts in this perspective: 15234 (31.8% of total)
Credibility Score: 73.2/100

Key themes: news, politics, election, vote, government

Summary:
This perspective focuses on political discussions about recent elections and 
government policy decisions. Posts emphasize the importance of voting and civic 
participation.

Engagement Statistics:
  • Average engagement: 0.45
  • Maximum engagement: 5628
  • Posts with engagement: 8934

Representative Posts:

  Example 1:
    "Political engagement is crucial in a democracy. Every vote counts..."
    [Engagement: 234, Platform: reddit]
```

## 🔍 Advanced Features

### Auto-Detect Optimal Clusters

Let the system determine the best number of clusters automatically:

```python
config['auto_detect_clusters'] = True
# Use max_clusters parameter to test up to K clusters
```

### Custom Embeddings Model

Use different SentenceTransformer models:

```python
config['embedding_model'] = 'all-mpnet-base-v2'  # More powerful, slower
config['embedding_model'] = 'all-MiniLM-L6-v2'  # Fast, good quality (default)
config['embedding_model'] = 'all-distilroberta-v1'  # Fast, lightweight
```

### Adaptive Field Detection

The system automatically detects and maps:

| Expected Field | Detected Alternatives |
|---|---|
| `text` | body, text, content, message, tweet |
| `engagement` | score, likes, upvotes, favorites |
| `replies` | num_comments, replies, comments |
| `timestamp` | created_utc, timestamp, date |
| `platform` | subreddit, platform, source, forum |
| `author` | author, username, user |

Missing fields are handled gracefully with defaults or NaN values.

## 📈 Performance Metrics

### Scalability

- **Small datasets** (< 10K posts): ~2-5 minutes
- **Medium datasets** (10K-100K): ~10-30 minutes
- **Large datasets** (> 100K): 30+ minutes (embedding generation bottleneck)

### Embeddings Optimization

For faster processing with large datasets:
```python
# Use lighter model
config['embedding_model'] = 'sentence-transformers/all-MiniLM-L6-v2'

# Or reduce embedding batch size
# (Edit clustering.py generate_embeddings method)
```

## 🛠️ Troubleshooting

### Issue: "sentence-transformers not installed"

```bash
pip install sentence-transformers
```

### Issue: "transformers not installed" (for abstractive summarization)

```bash
pip install transformers torch
```

### Issue: Out of memory with large embeddings

```python
# Sample the data instead of processing all
config['sample_size'] = 50000

# Or use a simpler clustering method
# (Modify clustering.py to use Agglomerative Clustering instead of KMeans)
```

### Issue: Very slow processing

1. Use a smaller sample: `config['sample_size'] = 10000`
2. Use lighter embedding model: `config['embedding_model'] = 'all-MiniLM-L6-v2'`
3. Reduce number of clusters: `config['n_clusters'] = 3`

## 🔐 Data Privacy & Ethics

The system is designed with privacy in mind:

- **No external calls**: All processing is local (except model downloads)
- **No data retention**: Processed data can be deleted after digest generation
- **Transparent scoring**: All credibility factors are documented
- **Multiple perspectives**: System intentionally shows different viewpoints
- **Source attribution**: All example posts include platform information

## 📚 Extending the System

### Add Custom Credibility Factors

Edit `credibility.py` and add scoring methods:

```python
def score_custom_factor(self, text: str) -> float:
    # Your custom scoring logic
    return score
```

### Implement Different Clustering

Replace KMeans in `clustering.py`:

```python
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=k)
labels = clustering.fit_predict(embeddings_scaled)
```

### Add Custom Summarization

Modify `summarization.py` to use different models:

```python
pipeline = pipeline("summarization", model="your-custom-model")
```

## 📝 References

**Libraries Used**:
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [scikit-learn](https://scikit-learn.org/) - Clustering & ML
- [SentenceTransformers](https://www.sbert.net/) - Embeddings
- [Hugging Face Transformers](https://huggingface.co/transformers/) - NLP models

**Models**:
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Summarization: `facebook/bart-large-cnn`

**Papers & Algorithms**:
- KMeans clustering
- Sentence embeddings (Reimers & Gurevych, 2019)
- BART summarization (Lewis et al., 2019)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Suggestions for improvements:

1. **Better toxicity detection**: Integrate perspective API or better classifiers
2. **Account credibility**: Add social media account analysis
3. **Temporal features**: Track how perspectives evolve over time
4. **Multi-language support**: Extend to non-English posts
5. **Real-time processing**: Adapt for streaming data
6. **Interactive visualization**: Create HTML dashboards
7. **Explainability**: Feature importance for credibility scores

## 📧 Questions?

For issues or questions about the system, refer to the module docstrings and inline comments for detailed explanations.

---

**Last Updated**: January 2024  
**Version**: 1.0.0
