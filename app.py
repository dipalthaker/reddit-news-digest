import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import re
import time
from datetime import datetime
from urllib.parse import urlparse
from html import unescape
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="Reddit News Digest",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0rem !important;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 3.2rem;
        padding-bottom: 2.5rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
        border-right: 1px solid #dde7f2;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 3.8rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 1.4rem;
    }

    .sidebar-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 1.15rem 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        margin-bottom: 1.15rem;
        backdrop-filter: blur(8px);
    }

    .sidebar-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
    }

    .sidebar-subtitle {
        font-size: 0.92rem;
        color: #64748b;
        line-height: 1.45;
        margin-bottom: 0.85rem;
    }

    .sidebar-pill {
        display: inline-block;
        padding: 0.3rem 0.72rem;
        border-radius: 999px;
        background: #eaf2ff;
        color: #1d4ed8;
        font-size: 0.72rem;
        font-weight: 700;
        border: 1px solid #c7dcff;
    }

    .sidebar-section-label {
        font-size: 0.76rem;
        font-weight: 800;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        margin-top: 0.35rem;
    }

    .sidebar-footer {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 0.85rem 0.95rem;
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.45;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
    }

    section[data-testid="stSidebar"] .stRadio > div {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 0.45rem 0.5rem 0.2rem 0.5rem;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
    }

    section[data-testid="stSidebar"] .stSelectbox > div[data-baseweb="select"] > div {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #d9e3ef;
        min-height: 48px;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.03);
    }

    section[data-testid="stSidebar"] .stTextInput > div > div input {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #d9e3ef;
        min-height: 46px;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.03);
    }

    section[data-testid="stSidebar"] .stTextInput > div > div input:focus {
        border-color: #93c5fd;
        box-shadow: 0 0 0 1px #93c5fd;
    }

    section[data-testid="stSidebar"] .stSlider {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0.1rem 0 0.2rem 0;
        box-shadow: none;
        margin-bottom: 0.5rem;
    }

    section[data-testid="stSidebar"] .stSlider label {
        font-weight: 600;
        color: #334155;
    }

    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 16px;
        font-weight: 700;
        min-height: 48px;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18);
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        color: white;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #334155;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetric"] label {
        color: #475569;
        font-size: 0.86rem;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
    }

    .hero {
        background: radial-gradient(circle at top left, #3b82f6 0%, #1e3a8a 42%, #312e81 100%);
        color: white;
        padding: 2rem;
        border-radius: 26px;
        box-shadow: 0 20px 45px rgba(30, 58, 138, 0.22);
        margin-bottom: 1.2rem;
    }

    .hero-chip {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #e0f2fe;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 850;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: 0.55rem 0 0 0;
        color: #e2e8f0;
        font-size: 0.98rem;
    }

    .keyword-chip {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: #dbeafe;
        color: #1e3a8a;
        font-size: 0.82rem;
        font-weight: 650;
        margin-right: 0.4rem;
        margin-bottom: 0.35rem;
        border: 1px solid #bfdbfe;
    }

    .soft-card {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.15rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        margin-bottom: 1rem;
    }

    .perspective-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.055);
        margin-bottom: 1.1rem;
    }

    .perspective-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .muted {
        color: #64748b;
        font-size: 0.9rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 850;
        color: #111827;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }

    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 750;
        margin-right: 0.45rem;
    }

    .badge-high {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    .badge-moderate {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }

    .badge-low {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    .badge-neutral {
        background: #e2e8f0;
        color: #334155;
        border: 1px solid #cbd5e1;
    }

    .warning-card {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .info-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .success-card {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .small-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }

    .viz-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 1rem 1rem 0.6rem 1rem;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
        height: 100%;
    }

    .viz-card h4 {
        margin: 0 0 0.25rem 0;
        font-size: 1rem;
        font-weight: 800;
        color: #0f172a;
    }

    .viz-card p {
        margin: 0 0 0.75rem 0;
        color: #64748b;
        font-size: 0.88rem;
    }

    .viz-section {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .viz-note {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }

    .chart-wrap {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        margin-bottom: 1rem;
    }

    div[data-testid="stTabs"] {
        margin-top: 0.75rem;
    }

    div[data-testid="stTabs"] > div {
        overflow: visible !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
        padding-left: 0.1rem;
        background: transparent;
        overflow-x: auto;
        overflow-y: visible !important;
        scrollbar-width: none;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto !important;
        min-height: 44px;
        background: #ffffff;
        border-radius: 999px;
        padding: 0.6rem 1.1rem;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
        white-space: nowrap;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
        border: 1px solid #93c5fd !important;
        color: #1d4ed8 !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.12);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: #2563eb !important;
        height: 3px !important;
        border-radius: 999px;
        bottom: -2px !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


STOPWORDS = {
    "what", "do", "does", "did", "people", "think", "about", "the", "a", "an",
    "is", "are", "was", "were", "to", "for", "of", "in", "on", "with", "and",
    "or", "this", "that", "these", "those", "how", "why", "when", "where",
    "who", "news", "discussion", "reddit", "posts", "post", "comments"
}


ALIAS_MAP = {
    "ai": ["ai", "artificial intelligence", "machine learning", "automation", "chatgpt", "openai"],
    "artificial intelligence": ["artificial intelligence", "ai", "machine learning", "automation", "chatgpt", "openai"],
    "masters": ["masters degree", "master degree", "master's degree", "graduate school", "grad school", "university", "college", "tuition", "students", "education"],
    "master": ["masters degree", "master degree", "master's degree", "graduate school", "grad school", "university", "college", "tuition", "students", "education"],
    "degree": ["degree", "graduate school", "grad school", "college", "university", "tuition", "students", "education"],
    "jobs": ["jobs", "layoffs", "salary", "workers", "employment", "unemployment", "paycheck", "hiring"],
    "job": ["jobs", "layoffs", "salary", "workers", "employment", "unemployment", "paycheck", "hiring"],
    "climate": ["climate", "weather", "storm", "heat", "flooding", "emissions", "wildfire"],
    "weather": ["weather", "storm", "rain", "flooding", "heat", "snow"],
    "election": ["election", "voting", "candidate", "campaign", "policy", "government", "senate", "congress"],
    "health": ["health", "hospital", "medicine", "disease", "vaccine", "doctor", "care"],
    "war": ["war", "military", "conflict", "ceasefire", "gaza", "israel", "ukraine", "russia"],
    "economy": ["economy", "inflation", "market", "prices", "wages", "jobs", "recession"],
    "money": ["money", "rent", "paycheck", "salary", "debt", "cost of living", "expenses", "inflation"]
}


AMBIGUOUS_REPLACE_ONLY = {"masters", "master"}


def clean_display_text(value):
    if value is None:
        return ""
    text = str(value)
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value


def extract_domain(url):
    try:
        if not isinstance(url, str) or not url.strip():
            return ""
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return ""


def find_csv_files():
    files = glob.glob(os.path.join(BASE_DIR, "*.csv"))
    ranked = []

    for path in files:
        name = os.path.basename(path).lower()
        score = 0

        if "reddit_news" in name:
            score += 100
        if "news" in name:
            score += 60
        if "reddit" in name:
            score += 40
        if "kaggle_rc" in name:
            score += 10

        ranked.append((score, path))

    ranked.sort(reverse=True)
    return [p for _, p in ranked]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_column(df, candidates):
    lower_map = {str(c).lower().strip(): c for c in df.columns}

    for c in candidates:
        if c in df.columns:
            return c

    for c in candidates:
        key = c.lower().strip()
        if key in lower_map:
            return lower_map[key]

    return None


def prepare_reddit_news_df(df):
    working = df.copy()

    title_col = detect_column(working, ["title", "headline", "post_title"])
    body_col = detect_column(working, ["body", "selftext", "text", "content", "description"])
    subreddit_col = detect_column(working, ["subreddit", "community"])
    score_col = detect_column(working, ["score", "upvotes", "ups"])
    comments_col = detect_column(working, ["num_comments", "comments", "comment_count"])
    url_col = detect_column(working, ["url", "link", "permalink"])
    date_col = detect_column(working, ["created_utc", "created", "date", "timestamp"])

    if title_col is None and body_col is None:
        raise ValueError(
            "Could not find a usable text column. The dataset needs title, headline, body, selftext, text, or content."
        )

    if title_col:
        working["title"] = working[title_col].fillna("").astype(str)
    elif body_col:
        working["title"] = working[body_col].fillna("").astype(str).str.slice(0, 110)
    else:
        working["title"] = ""

    working["body"] = working[body_col].fillna("").astype(str) if body_col else ""
    working["subreddit"] = working[subreddit_col].fillna("").astype(str) if subreddit_col else "unknown"
    working["url"] = working[url_col].fillna("").astype(str) if url_col else ""
    working["created_utc"] = working[date_col] if date_col else ""

    if score_col:
        working["score"] = pd.to_numeric(working[score_col], errors="coerce").fillna(0)
    else:
        working["score"] = 0

    if comments_col:
        working["num_comments"] = pd.to_numeric(working[comments_col], errors="coerce").fillna(0)
    else:
        working["num_comments"] = 0

    working["domain"] = working["url"].apply(extract_domain)

    working["analysis_text"] = (
        working["title"].fillna("").astype(str) + " " +
        working["body"].fillna("").astype(str)
    ).apply(clean_display_text)

    working["analysis_text_lower"] = (
        working["title"].fillna("").astype(str) + " " +
        working["body"].fillna("").astype(str) + " " +
        working["subreddit"].fillna("").astype(str) + " " +
        working["domain"].fillna("").astype(str)
    ).str.lower()

    working = working.drop_duplicates(subset=["title", "url"], keep="first")
    working = working[working["analysis_text"].astype(str).str.len() >= 20].copy()
    working = working.reset_index(drop=True)

    return working


def extract_keywords(user_text):
    if not user_text:
        return []

    text = str(user_text).strip().lower()

    if "," in text:
        pieces = [p.strip() for p in text.split(",")]
        keywords = [p for p in pieces if len(p) >= 2]
    else:
        words = re.findall(r"[a-zA-Z][a-zA-Z'\-]+", text)
        keywords = []
        for w in words:
            w = w.strip().lower()
            if len(w) >= 2 and w not in STOPWORDS:
                keywords.append(w)

    cleaned = []
    for kw in keywords:
        kw = kw.strip().lower().replace("’", "'")
        if kw and kw not in cleaned:
            cleaned.append(kw)

    return cleaned


def expand_keywords(keywords):
    expanded = []

    for kw in keywords:
        kw = kw.lower().strip()

        if kw in ALIAS_MAP:
            if kw not in AMBIGUOUS_REPLACE_ONLY:
                expanded.append(kw)
            expanded.extend(ALIAS_MAP[kw])
        else:
            expanded.append(kw)

    final = []
    for item in expanded:
        item = item.strip().lower()
        if item and item not in final:
            final.append(item)

    return final


def keyword_match(text, keyword):
    text = str(text).lower()
    keyword = str(keyword).lower().strip()

    if not keyword:
        return False

    if " " in keyword:
        return keyword in text

    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def score_row_relevance(row, expanded_keywords):
    title = str(row.get("title", "")).lower()
    body = str(row.get("body", "")).lower()
    subreddit = str(row.get("subreddit", "")).lower()
    domain = str(row.get("domain", "")).lower()

    score = 0.0
    matched_terms = []

    for kw in expanded_keywords:
        title_match = keyword_match(title, kw)
        body_match = keyword_match(body, kw)
        subreddit_match = keyword_match(subreddit, kw)
        domain_match = keyword_match(domain, kw)

        if title_match:
            score += 4.0
            matched_terms.append(kw)

        if body_match:
            score += 2.0
            matched_terms.append(kw)

        if subreddit_match:
            score += 1.0
            matched_terms.append(kw)

        if domain_match:
            score += 0.5
            matched_terms.append(kw)

    return score, ", ".join(sorted(set(matched_terms)))


def filter_posts_by_relevance(df, user_keywords, min_relevance=2.0, max_rows=500):
    if df is None or df.empty:
        return df, 0, 0, []

    working = df.copy()
    expanded_keywords = expand_keywords(user_keywords)

    if not expanded_keywords:
        return working.iloc[0:0], 0, len(working), []

    scores = working.apply(
        lambda row: score_row_relevance(row, expanded_keywords),
        axis=1
    )

    working["relevance_score"] = [s[0] for s in scores]
    working["matched_terms"] = [s[1] for s in scores]

    weak_preview = (
        working[working["relevance_score"] > 0]
        .sort_values(["relevance_score", "score", "num_comments"], ascending=[False, False, False])
        .head(5)["title"]
        .astype(str)
        .tolist()
    )

    filtered = working[working["relevance_score"] >= min_relevance].copy()

    if filtered.empty:
        return filtered, 0, len(working), weak_preview

    filtered = filtered.sort_values(
        ["relevance_score", "score", "num_comments"],
        ascending=[False, False, False]
    )

    filtered = filtered.head(max_rows).reset_index(drop=True)

    return filtered, len(filtered), len(working), weak_preview


def quality_filter_posts(df, min_text_chars=20):
    if df is None or df.empty:
        return df, 0, 0

    working = df.copy()
    before = len(working)

    bad_patterns = [
        r"\[deleted\]",
        r"\[removed\]",
        r"automoderator",
        r"your submission has been removed",
        r"megathread",
        r"daily discussion thread",
        r"weekly discussion thread"
    ]

    bad_regex = "|".join(bad_patterns)

    working = working[
        ~working["analysis_text"].astype(str).str.lower().str.contains(
            bad_regex,
            regex=True,
            na=False
        )
    ].copy()

    working = working[working["analysis_text"].astype(str).str.len() >= min_text_chars].copy()

    after = len(working)

    return working.reset_index(drop=True), before, after


def compute_credibility(df):
    working = df.copy()

    score_signal = np.log1p(pd.to_numeric(working["score"], errors="coerce").fillna(0))
    comment_signal = np.log1p(pd.to_numeric(working["num_comments"], errors="coerce").fillna(0))

    if score_signal.max() > score_signal.min():
        score_norm = (score_signal - score_signal.min()) / (score_signal.max() - score_signal.min())
    else:
        score_norm = pd.Series(0.5, index=working.index)

    if comment_signal.max() > comment_signal.min():
        comment_norm = (comment_signal - comment_signal.min()) / (comment_signal.max() - comment_signal.min())
    else:
        comment_norm = pd.Series(0.5, index=working.index)

    title_len = working["title"].astype(str).str.len()
    body_len = working["body"].astype(str).str.len()

    text_quality = (
        np.clip(title_len / 90, 0, 1) * 0.65 +
        np.clip(body_len / 300, 0, 1) * 0.35
    )

    credible_domains = [
        "reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "npr.org",
        "nytimes.com", "washingtonpost.com", "theguardian.com", "cnn.com",
        "nbcnews.com", "cbsnews.com", "abcnews.go.com", "politico.com",
        "thehill.com", "axios.com", "bloomberg.com"
    ]

    domain_score = working["domain"].astype(str).apply(
        lambda d: 1.0 if any(cd in d for cd in credible_domains) else 0.65 if d else 0.55
    )

    text_lower = working["analysis_text_lower"].astype(str)

    toxic_or_misinfo = text_lower.str.contains(
        r"\b(fake|hoax|conspiracy|sheeple|traitor|rigged|scam|propaganda|idiot|stupid|kill|destroy)\b",
        regex=True,
        na=False
    ).astype(float)

    credibility = (
        35 +
        score_norm * 18 +
        comment_norm * 15 +
        text_quality * 16 +
        domain_score * 12 -
        toxic_or_misinfo * 10
    )

    working["credibility_score"] = np.clip(credibility, 0, 100).round(2)

    def level(score):
        if score >= 75:
            return "High"
        if score >= 55:
            return "Moderate"
        if score >= 35:
            return "Low"
        return "Very Low"

    working["credibility_level"] = working["credibility_score"].apply(level)

    return working


def run_clustering(df, n_clusters):
    working = df.copy()
    texts = working["analysis_text"].fillna("").astype(str).tolist()

    if len(working) == 0:
        raise ValueError("No records available for clustering.")

    n_clusters = max(1, min(int(n_clusters), len(working)))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=4000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    matrix = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    if n_clusters == 1 or len(working) < 3:
        working["cluster"] = 0
        summed = np.asarray(matrix.sum(axis=0)).ravel()
        top_idx = summed.argsort()[::-1][:10]
        keyword_map = {0: terms[top_idx].tolist()}
        return working, keyword_map

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    working["cluster"] = labels

    keyword_map = {}
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    for cluster_id in range(n_clusters):
        top_terms = terms[order_centroids[cluster_id, :10]].tolist()
        keyword_map[cluster_id] = top_terms

    return working, keyword_map


def summarize_cluster(cluster_df, keywords, topic):
    sorted_df = cluster_df.sort_values(
        ["relevance_score", "score", "num_comments", "credibility_score"],
        ascending=[False, False, False, False]
    )

    top_titles = sorted_df["title"].dropna().astype(str).head(3).tolist()
    top_subreddits = sorted_df["subreddit"].dropna().astype(str).value_counts().head(2).index.tolist()
    top_domains = sorted_df["domain"].dropna().astype(str).replace("", np.nan).dropna().value_counts().head(2).index.tolist()

    keyword_text = ", ".join(keywords[:4]) if keywords else topic

    summary = f"This group is mainly about {keyword_text}."

    if top_titles:
        clean_titles = [clean_display_text(t) for t in top_titles]
        summary += " The strongest records in this group mention: " + "; ".join(clean_titles) + "."

    if top_subreddits:
        summary += " Most of this discussion appears in " + ", ".join(top_subreddits) + "."

    if top_domains:
        summary += " Common linked sources include " + ", ".join(top_domains) + "."

    return summary


def credibility_level_from_score(score):
    try:
        score = float(score)
    except Exception:
        return "Unknown"

    if score >= 75:
        return "High"
    if score >= 55:
        return "Moderate"
    if score >= 35:
        return "Low"
    return "Very Low"


def make_executive_summary(topic, total, perspectives):
    if not perspectives:
        return "No strong discussion patterns were found for this topic."

    largest = perspectives[0]
    terms = ", ".join(largest.get("keywords", [])[:4])
    avg_cred = round(np.mean([p["credibility_score"] for p in perspectives]), 2)

    return (
        f"The digest analyzed {total} Reddit records related to {topic}. "
        f"The largest discussion group focuses on {terms}. "
        f"Across the discussion, the average credibility signal is {avg_cred}. "
        "The cards below explain the main discussion patterns rather than treating them as verified news conclusions."
    )


def build_digest(df, keyword_map, topic, original_keywords, source_name, before_quality, after_quality):
    perspectives = []
    total = len(df)

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id].copy()
        keywords = keyword_map.get(cluster_id, [])

        avg_cred = round(cluster_df["credibility_score"].mean(), 2)
        level = credibility_level_from_score(avg_cred)

        examples = []
        example_df = cluster_df.sort_values(
            ["relevance_score", "score", "num_comments"],
            ascending=[False, False, False]
        ).head(3)

        for _, row in example_df.iterrows():
            examples.append({
                "title": clean_display_text(row.get("title", "")),
                "body": clean_display_text(row.get("body", ""))[:350],
                "subreddit": clean_display_text(row.get("subreddit", "")),
                "score": float(row.get("score", 0)),
                "num_comments": float(row.get("num_comments", 0)),
                "url": clean_display_text(row.get("url", "")),
                "domain": clean_display_text(row.get("domain", "")),
                "credibility_score": float(row.get("credibility_score", 0)),
                "matched_terms": clean_display_text(row.get("matched_terms", ""))
            })

        perspectives.append({
            "cluster_id": int(cluster_id),
            "cluster_size": int(len(cluster_df)),
            "percentage_of_total": round((len(cluster_df) / total) * 100, 1) if total else 0,
            "credibility_score": avg_cred,
            "credibility_level": level,
            "keywords": [clean_display_text(k) for k in keywords[:8]],
            "summary": summarize_cluster(cluster_df, keywords, topic),
            "example_posts": examples
        })

    perspectives = sorted(
        perspectives,
        key=lambda p: (p["cluster_size"], p["credibility_score"]),
        reverse=True
    )

    for i, p in enumerate(perspectives, 1):
        p["rank"] = i

    metadata = {
        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_title": topic,
        "source": source_name,
        "search_keywords": original_keywords,
        "total_posts_analyzed": int(total),
        "num_perspectives": len(perspectives),
        "quality_filter_before": int(before_quality),
        "quality_filter_after": int(after_quality)
    }

    return {
        "metadata": metadata,
        "executive_summary": make_executive_summary(topic, total, perspectives),
        "perspectives": perspectives
    }


def save_outputs(digest, df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    topic = digest.get("metadata", {}).get("event_title", "digest")
    safe_topic = normalize_name(topic)[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"digest_{safe_topic}_{ts}.json")
    csv_path = os.path.join(output_dir, f"data_{safe_topic}_{ts}.csv")
    txt_path = os.path.join(output_dir, f"digest_{safe_topic}_{ts}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2)

    df.to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(digest.get("executive_summary", "") + "\n\n")
        for p in digest.get("perspectives", []):
            f.write(f"Perspective {p.get('rank')}: {p.get('summary')}\n")

    return txt_path, json_path, csv_path


def load_latest_results():
    if not os.path.exists(OUTPUT_DIR):
        return None, None, {}

    json_files = glob.glob(os.path.join(OUTPUT_DIR, "digest_*.json"))
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, "data_*.csv"))
    txt_files = glob.glob(os.path.join(OUTPUT_DIR, "digest_*.txt"))

    if not json_files:
        return None, None, {}

    latest_json = max(json_files, key=os.path.getmtime)

    with open(latest_json, "r", encoding="utf-8") as f:
        digest = json.load(f)

    df = None
    latest_csv = None

    if csv_files:
        latest_csv = max(csv_files, key=os.path.getmtime)
        df = pd.read_csv(latest_csv)

    latest_txt = max(txt_files, key=os.path.getmtime) if txt_files else None

    return digest, df, {
        "json": latest_json,
        "csv": latest_csv,
        "txt": latest_txt
    }


def suggest_keywords(topic):
    text = str(topic).lower()

    if "master" in text or "degree" in text or "college" in text:
        return "- graduate school, degree, university, students, tuition, education"

    if "ai" in text or "automation" in text:
        return "- AI, artificial intelligence, automation, jobs, technology"

    if "job" in text or "layoff" in text:
        return "- jobs, layoffs, salary, workers, unemployment, paycheck"

    if "climate" in text or "weather" in text:
        return "- climate, weather, storm, flooding, heat"

    if "election" in text or "politic" in text:
        return "- election, voting, candidate, government, policy"

    return (
        "- AI, automation, jobs, technology\n"
        "- climate, weather, storm, flooding\n"
        "- election, voting, government, policy\n"
        "- graduate school, degree, university, students, tuition"
    )


def run_analysis(csv_path, topic, n_clusters, max_rows, min_required):
    raw_df = load_dataset(csv_path)
    prepared = prepare_reddit_news_df(raw_df)

    keywords = extract_keywords(topic)

    if not keywords:
        raise ValueError(
            "Please enter clearer keywords. Example: graduate school, degree, university, students, tuition, education"
        )

    filtered, match_count, total_count, weak_preview = filter_posts_by_relevance(
        prepared,
        keywords,
        min_relevance=2.0,
        max_rows=max_rows
    )

    if match_count < min_required:
        suggestions = suggest_keywords(topic)

        preview_text = ""
        if weak_preview:
            preview_text = "\n\nClosest weak matches found:\n" + "\n".join([f"- {t}" for t in weak_preview])

        raise ValueError(
            f"Not enough relevant posts were found for this topic.\n\n"
            f"Only {match_count} records passed the relevance filter out of {total_count} rows.\n\n"
            f"Try broader or clearer keywords:\n{suggestions}"
            f"{preview_text}"
        )

    filtered, before_quality, after_quality = quality_filter_posts(filtered)

    if after_quality < min_required:
        raise ValueError(
            f"Only {after_quality} usable records remained after quality filtering. "
            "Try broader keywords or a topic that is more common in this dataset."
        )

    scored = compute_credibility(filtered)
    clustered, keyword_map = run_clustering(scored, n_clusters)

    source_name = os.path.basename(csv_path)

    digest = build_digest(
        clustered,
        keyword_map,
        topic,
        keywords,
        source_name,
        before_quality,
        after_quality
    )

    txt_path, json_path, csv_path_out = save_outputs(digest, clustered, OUTPUT_DIR)

    files = {
        "txt": txt_path,
        "json": json_path,
        "csv": csv_path_out
    }

    return digest, clustered, files


def badge_class(level):
    level = str(level).lower()

    if "high" in level:
        return "badge badge-high"
    if "moderate" in level:
        return "badge badge-moderate"
    if "low" in level:
        return "badge badge-low"

    return "badge badge-neutral"


def render_hero(digest):
    metadata = digest.get("metadata", {})
    topic = metadata.get("event_title", "Reddit News Digest")
    generated = metadata.get("generated_date", "N/A")
    source = metadata.get("source", "N/A")
    keywords = metadata.get("search_keywords", [])

    chips = ""
    for kw in keywords:
        chips += f"<span class='keyword-chip'>{clean_display_text(kw)}</span>"

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-chip">Human centered social media digest</div>
            <h1>Daily Reddit News Digest</h1>
            <p><strong>{clean_display_text(topic)}</strong></p>
            <p>Generated on {clean_display_text(generated)} · Source: {clean_display_text(source)}</p>
            <div style="margin-top:1rem;">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metrics(digest, df):
    metadata = digest.get("metadata", {})
    perspectives = digest.get("perspectives", [])

    total = metadata.get("total_posts_analyzed", len(df) if df is not None else 0)
    n_perspectives = metadata.get("num_perspectives", len(perspectives))

    avg_cred = "N/A"
    high_count = "N/A"

    if df is not None and "credibility_score" in df.columns:
        scores = pd.to_numeric(df["credibility_score"], errors="coerce").dropna()
        if not scores.empty:
            avg_cred = round(scores.mean(), 2)
            high_count = int((scores >= 75).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts Analyzed", total)
    c2.metric("Perspectives", n_perspectives)
    c3.metric("Avg Credibility", avg_cred)
    c4.metric("High Credibility Posts", high_count)


def render_digest_tab(digest, df, files):
    render_hero(digest)
    render_metrics(digest, df)

    metadata = digest.get("metadata", {})
    before = metadata.get("quality_filter_before")
    after = metadata.get("quality_filter_after")

    if before is not None and after is not None:
        st.markdown(
            f"""
            <div class="info-card">
            Quality filter retained <strong>{after}</strong> of <strong>{before}</strong> relevant records.
            If this number is very small, the dataset probably does not contain enough posts about the searched topic.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="info-card">
        <strong>What this shows</strong><br>
        Each card below is one group of similar Reddit records. The summary explains what that group is mainly about.
        It is a discussion pattern, not a verified news conclusion.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)
    st.write(digest.get("executive_summary", "No summary available."))

    if files:
        with st.expander("Output files"):
            for key, value in files.items():
                if value:
                    st.write(f"{key.upper()}: {value}")

    st.markdown("<div class='section-title'>What the Reddit discussion is saying</div>", unsafe_allow_html=True)

    perspectives = digest.get("perspectives", [])

    if not perspectives:
        st.warning("No perspectives were created.")
        return

    for p in perspectives:
        render_perspective_card(p)


def render_perspective_card(p):
    rank = p.get("rank", "")
    cluster_id = p.get("cluster_id", "")
    keywords = [clean_display_text(k) for k in p.get("keywords", []) if clean_display_text(k)]
    credibility = p.get("credibility_score", "N/A")
    level = p.get("credibility_level", "N/A")
    size = p.get("cluster_size", "N/A")
    pct = p.get("percentage_of_total", "N/A")
    summary = clean_display_text(p.get("summary", ""))

    title_terms = ", ".join(keywords[:3]) if keywords else "related records"

    chips = ""
    for kw in keywords[:8]:
        chips += f"<span class='keyword-chip'>{kw}</span>"

    badge = badge_class(level)

    st.markdown(
        f"""
        <div class="perspective-card">
            <div class="perspective-title">Perspective {rank}: {title_terms}</div>
            <div class="muted">Cluster {cluster_id} · {size} records · {pct}% of matched discussion</div>
            <div style="margin-top:0.8rem;">
                <span class="{badge}">{clean_display_text(level)}</span>
                <span class="muted">Credibility score: <strong>{credibility}</strong></span>
            </div>
            <div style="margin-top:1rem;">{chips}</div>
            <div style="margin-top:1rem;" class="small-label">What this group is mainly about</div>
            <div>{summary}</div>
            <div style="margin-top:1rem;" class="small-label">How to read this</div>
            <div class="muted">
                This is a cluster of similar Reddit records. The credibility score is a heuristic signal based on
                engagement, comments, source domain, text quality, and misinformation or toxicity indicators.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    examples = p.get("example_posts", [])

    if examples:
        with st.expander(f"Representative records in Perspective {rank}"):
            for i, ex in enumerate(examples, 1):
                title = clean_display_text(ex.get("title", ""))
                subreddit = clean_display_text(ex.get("subreddit", ""))
                domain = clean_display_text(ex.get("domain", ""))
                score = ex.get("score", 0)
                comments = ex.get("num_comments", 0)
                url = clean_display_text(ex.get("url", ""))

                st.markdown(f"**{i}. {title}**")
                st.caption(f"r/{subreddit} · {domain} · score {score} · comments {comments}")
                if url:
                    st.write(url)


def top_n_with_other(series, n=5):
    counts = (
        series.astype(str)
        .replace("", np.nan)
        .replace("nan", np.nan)
        .dropna()
        .value_counts()
    )

    if counts.empty:
        return counts

    if len(counts) <= n:
        return counts

    top = counts.head(n).copy()
    other_sum = counts.iloc[n:].sum()

    if other_sum > 0:
        top.loc["Other"] = other_sum

    return top


def make_donut_chart(values, labels, title):
    if values is None or len(values) == 0 or np.sum(values) == 0:
        return None

    palette = [
        "#2563eb", "#60a5fa", "#7c3aed", "#818cf8",
        "#0ea5e9", "#14b8a6", "#f59e0b", "#ef4444"
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        startangle=90,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
        pctdistance=0.78,
        colors=palette[:len(values)],
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2)
    )

    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(centre_circle)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("equal")

    legend_labels = [f"{lab} ({val})" for lab, val in zip(labels, values)]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=9
    )

    plt.tight_layout()
    return fig


def render_chart_card(title, subtitle, values, labels):
    st.markdown(
        f"""
        <div class="viz-card">
            <h4>{title}</h4>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    fig = make_donut_chart(values, labels, title)

    if fig is None:
        st.info("Not enough data for this chart.")
    else:
        st.pyplot(fig)
        plt.close(fig)


def render_visualizations(digest, df):
    if df is None or df.empty:
        st.warning("No data available for charts.")
        return

    st.markdown("<div class='section-title'>Visual Summary</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="viz-note">
        A cleaner view of how the matched Reddit records are distributed across clusters,
        credibility levels, subreddits, and source domains.
        </div>
        """,
        unsafe_allow_html=True
    )

    score_col = "credibility_score" if "credibility_score" in df.columns else None
    level_col = "credibility_level" if "credibility_level" in df.columns else None
    cluster_col = "cluster" if "cluster" in df.columns else None
    subreddit_col = "subreddit" if "subreddit" in df.columns else None
    domain_col = "domain" if "domain" in df.columns else None

    valid_scores = pd.Series(dtype=float)

    if score_col:
        valid_scores = pd.to_numeric(df[score_col], errors="coerce").dropna()

    max_score = round(valid_scores.max(), 2) if not valid_scores.empty else "N/A"
    avg_score = round(valid_scores.mean(), 2) if not valid_scores.empty else "N/A"
    high_cred = int((valid_scores >= 75).sum()) if not valid_scores.empty else 0
    moderate_plus = int((valid_scores >= 55).sum()) if not valid_scores.empty else 0

    a, b, c, d = st.columns(4)
    a.metric("Average Credibility", avg_score)
    b.metric("Highest Score", max_score)
    c.metric("High Credibility Records", high_cred)
    d.metric("Moderate or Better", moderate_plus)

    st.markdown("<div class='viz-section'>", unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        if cluster_col:
            cluster_counts = (
                df[cluster_col]
                .astype(str)
                .value_counts()
                .sort_index()
            )
            render_chart_card(
                "Records by Perspective",
                "How the matched records are distributed across clusters.",
                cluster_counts.values,
                [f"Cluster {x}" for x in cluster_counts.index]
            )
        else:
            st.info("Cluster information is not available.")

    with row1_col2:
        if level_col:
            cred_counts = df[level_col].astype(str).value_counts()
            render_chart_card(
                "Credibility Breakdown",
                "Share of records by credibility level.",
                cred_counts.values,
                cred_counts.index.tolist()
            )
        else:
            st.info("Credibility level data is not available.")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        if subreddit_col:
            subreddit_counts = top_n_with_other(df[subreddit_col], n=5)
            render_chart_card(
                "Top Subreddits",
                "Where most matched records are coming from.",
                subreddit_counts.values,
                subreddit_counts.index.tolist()
            )
        else:
            st.info("Subreddit data is not available.")

    with row2_col2:
        if domain_col:
            domain_counts = top_n_with_other(df[domain_col], n=5)

            if not domain_counts.empty:
                render_chart_card(
                    "Top Linked Domains",
                    "Most common linked sources in the matched records.",
                    domain_counts.values,
                    domain_counts.index.tolist()
                )
            else:
                st.info("No domain/source links are available for this dataset.")
        else:
            st.info("Domain data is not available.")

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        if score_col and cluster_col and not valid_scores.empty:
            high_df = df[pd.to_numeric(df[score_col], errors="coerce") >= 75].copy()

            if not high_df.empty:
                high_counts = (
                    high_df[cluster_col]
                    .astype(str)
                    .value_counts()
                    .sort_index()
                )
                render_chart_card(
                    "High Credibility by Cluster",
                    "Which clusters contain the strongest credibility signals.",
                    high_counts.values,
                    [f"Cluster {x}" for x in high_counts.index]
                )
            else:
                st.info("No records reached the high credibility threshold.")
        else:
            st.info("Not enough data for high-credibility cluster analysis.")

    with row3_col2:
        if score_col and cluster_col and not valid_scores.empty:
            max_val = valid_scores.max()
            max_df = df[pd.to_numeric(df[score_col], errors="coerce") == max_val].copy()

            if not max_df.empty:
                max_counts = (
                    max_df[cluster_col]
                    .astype(str)
                    .value_counts()
                    .sort_index()
                )
                render_chart_card(
                    "Maximum Score by Cluster",
                    "Where the very highest-credibility records are concentrated.",
                    max_counts.values,
                    [f"Cluster {x}" for x in max_counts.index]
                )
            else:
                st.info("No maximum-score records were found.")
        else:
            st.info("Not enough data for max-score analysis.")

    st.markdown("</div>", unsafe_allow_html=True)

    if "perspectives" in digest and digest["perspectives"]:
        st.markdown("<div class='section-title'>Perspective Snapshot</div>", unsafe_allow_html=True)

        rows = []
        for p in digest["perspectives"]:
            rows.append({
                "Perspective": f"Perspective {p.get('rank', '')}",
                "Cluster": p.get("cluster_id", ""),
                "Records": p.get("cluster_size", ""),
                "Share %": p.get("percentage_of_total", ""),
                "Credibility": p.get("credibility_score", ""),
                "Level": p.get("credibility_level", ""),
                "Keywords": ", ".join(p.get("keywords", [])[:4])
            })

        p_df = pd.DataFrame(rows)
        st.dataframe(p_df, width="stretch", hide_index=True)


def render_data_preview(df):
    if df is None or df.empty:
        st.warning("No data available.")
        return

    st.markdown("<div class='section-title'>Data Preview</div>", unsafe_allow_html=True)

    preview_cols = [
        "title", "subreddit", "score", "num_comments", "domain",
        "relevance_score", "matched_terms", "cluster", "credibility_score", "credibility_level"
    ]

    available = [c for c in preview_cols if c in df.columns]
    st.dataframe(df[available].head(150), width="stretch")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download analyzed data",
        csv_data,
        file_name="reddit_digest_results.csv",
        mime="text/csv"
    )


csv_files = find_csv_files()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Reddit News Digest</div>
            <div class="sidebar-subtitle">
                Human centered discussion analysis for topic-based Reddit news exploration.
            </div>
            <span class="sidebar-pill">HCDS Project</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Navigation",
        ["About", "Run Analysis", "Explore Results"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)

    selected_csv = None

    if csv_files:
        st.markdown('<div class="sidebar-section-label">Dataset</div>', unsafe_allow_html=True)
        csv_names = [os.path.basename(f) for f in csv_files]
        selected_name = st.selectbox(
            "Dataset",
            csv_names,
            label_visibility="collapsed"
        )
        selected_csv = csv_files[csv_names.index(selected_name)]
    else:
        st.error("No CSV dataset found in the project folder.")

    if mode == "Run Analysis":
        st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Topic Setup</div>', unsafe_allow_html=True)

        topic = st.text_input(
            "Search topic or keywords",
            value="",
            placeholder="graduate school, degree, university, students, tuition",
            label_visibility="collapsed"
        )

        st.caption("Use several specific keywords for better matching.")

        n_clusters = st.slider("Perspectives", 1, 6, 3)
        max_rows = st.slider("Maximum matched records", 50, 1000, 500, step=50)
        min_required = st.slider("Minimum relevant records required", 2, 25, 5)

        st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

        run_button = st.button("Run Analysis", type="primary", width="stretch")
    else:
        topic = ""
        n_clusters = 3
        max_rows = 500
        min_required = 5
        run_button = False

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-footer">
            <strong>Group 3</strong><br>
            HCDS Spring 2026
        </div>
        """,
        unsafe_allow_html=True
    )


if mode == "About":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-chip">Project overview</div>
            <h1>Personalized Reddit News Digest</h1>
            <p>A dashboard that turns Reddit records into readable discussion summaries.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="soft-card">
        <h3>What this app does</h3>
        <p>
        The app searches a Reddit dataset, keeps only records that strongly match the selected topic,
        groups similar records into perspectives, scores credibility signals, and displays the result as a readable digest.
        </p>
        </div>

        <div class="soft-card">
        <h3>Why relevance filtering matters</h3>
        <p>
        A single keyword can be misleading. For example, searching <strong>Masters</strong> can match
        <strong>Blake Masters</strong>, the politician, instead of master's degrees or graduate school.
        This version uses stricter matching and rejects weak results instead of generating an inaccurate digest.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Better search examples")

    st.code(
        """graduate school, degree, university, students, tuition, education
AI, artificial intelligence, automation, jobs, technology
climate, weather, storm, flooding, heat
election, voting, candidate, government, policy
jobs, layoffs, salary, workers, unemployment, paycheck""",
        language="text"
    )


elif mode == "Run Analysis":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-chip">Run pipeline</div>
            <h1>Create a Digest</h1>
            <p>Choose a dataset, enter a focused topic, then generate clustered discussion summaries.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if selected_csv is None:
        st.stop()

    if not topic.strip():
        st.markdown(
            """
            <div class="info-card">
            Enter a topic in the sidebar. Use several clear keywords instead of one vague word.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    if run_button:
        try:
            progress = st.progress(0)
            status = st.empty()

            status.info("Loading and filtering dataset...")
            progress.progress(20)

            start = time.time()

            digest, result_df, files = run_analysis(
                selected_csv,
                topic,
                n_clusters,
                max_rows,
                min_required
            )

            progress.progress(100)
            elapsed = round(time.time() - start, 2)

            st.session_state["current_digest"] = digest
            st.session_state["current_df"] = result_df
            st.session_state["current_files"] = files

            status.success(f"Analysis complete in {elapsed} seconds.")

            st.markdown(
                """
                <div class="success-card">
                Digest created successfully. Open Explore Results to review the summary and charts.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("Output files:")
            st.write(files)

        except Exception as e:
            st.error(str(e))

            st.markdown("### Try one of these examples")
            st.code(suggest_keywords(topic), language="text")


elif mode == "Explore Results":
    digest = st.session_state.get("current_digest")
    df = st.session_state.get("current_df")
    files = st.session_state.get("current_files", {})

    if digest is None:
        digest, df, files = load_latest_results()

    if digest is None:
        st.markdown(
            """
            <div class="warning-card">
            No saved analysis was found. Go to Run Analysis and create a digest first.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Digest", "Visualizations", "Data"])

    with tab1:
        render_digest_tab(digest, df, files)

    with tab2:
        render_visualizations(digest, df)

    with tab3:
        render_data_preview(df)