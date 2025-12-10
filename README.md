# Reddit Supplement Analysis

This project was created for text analysis classes. Its goal is to collect, clean, enrich, and analyze Reddit discussions related to supplements and nootropics, and then extract sentiment, perceived benefits, and insights from real user experiences.

---

## 📥 Data Collection

The dataset consists of **all Reddit submissions and comments (2005–present)** from the following subreddits:

* `MushroomSupplements`
* `NooTopics`
* `NootropicsDepot`
* `Preworkoutsupplements`
* `Supplements`

Instructions for downloading the raw data using Pushshift dumps can be found here:
**[https://www.reddit.com/r/pushshift/comments/1itme1k/separate_dump_files_for_the_top_40k_subreddits/](https://www.reddit.com/r/pushshift/comments/1itme1k/separate_dump_files_for_the_top_40k_subreddits/)**

For each subreddit, this process produces two files:

```
<subreddit>_submissions
<subreddit>_comments
```

Example:
`Supplements_submissions` and `Supplements_comments`

---

## 🔧 Step 1 – Joining Submissions & Comments (`join.py`)

The first processing step aggregates comments under their corresponding submissions.

**Output example (truncated):**

```json
{
  "8r91bv": {
    "archived": false,
    "author": "Kostya93",
    "created_utc": 1529046617,
    "title": "Bioavailability of Medicinal Mushroom extracts or Why Extraction Is Essential",
    "selftext": "There is a lot of bad and/or incomplete information ...",
    "subreddit": "MushroomSupplements",
    "comments": [
      {
        "author": "SEIGOF_KONN",
        "body": "I wonder how this would apply to psychedelic fungi...",
        "created_utc": 1536531968,
        "score": 3
      },
      ...
    ]
  }
}
```

This produces a consolidated dataset where each submission includes all of its comments.

---

## 🧠 Step 2 – Sentiment & Benefit Extraction (`sentiment_analysis.py`)

The next stage performs:

* Sentence-level sentiment classification
* Supplement name detection
* Benefit categorization
* Question/statement classification

The output file produced is:
**`data_for_analysis_final.json`**

**Example entry:**

```json
{
  "supplement_found": "Lions Mane",
  "classification": "positive",
  "sentiment_score": 0.6249,
  "is_question": false,
  "sentence_context": "Lions Mane is great for memory and focus.",
  "full_text_source": "Okay so you are looking for a boost in mental performance...",
  "doc_type": "comment",
  "doc_id": "jx72cyp",
  "author": "Warm_Science_8229",
  "subreddit": "MushroomSupplements",
  "submission_id": "15xale5",
  "perceived_benefit": "Focus / Cognition",
  "benefit_confidence_score": 0.6508632898330688
}
```

This enriched dataset forms the basis for subsequent quantitative and qualitative analysis.

---

## 📊 Step 3 – Analysis & Visualization (`analysis.ipynb`)

All final analytical steps — such as:

* Sentiment distributions
* Most discussed supplements
* Benefit–supplement relationships
* Temporal trends
* Visual plots

…are performed inside **`analysis.ipynb`**.

This notebook uses the previously generated JSON to produce insights and visual summaries.

