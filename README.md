# 🎨 Painting Similarity Search with DINO & Qdrant

This project is a visual similarity search engine for paintings, powered by the [DINO image model](https://ai.facebook.com/blog/dino-self-supervised-learning/) for embedding generation and [Qdrant](https://qdrant.tech) as the vector database. The app is built using Streamlit and supports both local and cloud deployment.

---

## 🚀 Features

- 🔍 Similarity search between artworks using DINO embeddings
- 🧠 Precomputed Qdrant snapshot included — no embedding required at runtime
---

## 📁 Project Structure

```

.
├── main.py                        # Streamlit app interface
├── dino\_embedding.ipynb           # Notebook for generating DINO embeddings
├── snapshots/
│   └── painting\_collection.snapshot   # Precomputed vector DB snapshot
├── .env.example                   # Template for local environment config
├── .gitignore
├── requirements.txt               # Python dependencies
└── README.md

````

---

## ⚙️ Requirements

- Python 3.8 or higher
- Dependencies: see `requirements.txt`

You can install them via:

```bash
pip install -r requirements.txt
````

---

## 🧪 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/painting-similarity-app.git
cd painting-similarity-app
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

Then fill in your value:

```env
QDRANT_API_KEY=your-api-key
```

And update these two in the other files:

QDRANT_HOST=https://your-cluster-xyz.qdrant.tech
COLLECTION_NAME=painting_collection

Get your free API key and cluster from [cloud.qdrant.io](https://cloud.qdrant.io).

---

## 💾 Restore the Qdrant Snapshot

The snapshot contains all the precomputed DINO embeddings.

### Option A – Qdrant Cloud UI

1. Go to your cluster
2. Create a collection named `painting_collection` (if it doesn’t exist)
3. Navigate to the **Snapshots** tab
4. Click **Recover from Snapshot**
5. Upload the file: `snapshots/painting_collection.snapshot`

---

## ▶️ Run the App Locally

```bash
streamlit run main.py
```

---

## 📓 Optional: Generate Embeddings with DINO

You can reprocess your own images using the `dino_embedding.ipynb` notebook:

```bash
jupyter notebook dino_embedding.ipynb
```

This notebook loads the DINO model and encodes images as vector embeddings to be inserted into Qdrant.


