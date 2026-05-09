import pickle
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "model/content_model.pkl"

def build_content_features(products):
    """Build TF-IDF feature matrix from product attributes."""
    df = pd.DataFrame(products)
    if df.empty:
        return None, None

    # Combine text features
    df['features'] = (
        df['name'].fillna('') + ' ' +
        df['category'].fillna('') + ' ' +
        df['description'].fillna('')
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=500,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(df['features'])

    # Save model
    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "vectorizer":   vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "product_ids":  df['product_id'].tolist()
        }, f)

    return vectorizer, tfidf_matrix

def load_content_model():
    """Load saved content-based model."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def get_similar_products(product_id, n=10):
    """Get products similar to a given product."""
    model = load_content_model()
    if not model:
        return []

    product_ids  = model['product_ids']
    tfidf_matrix = model['tfidf_matrix']

    if product_id not in product_ids:
        return []

    idx         = product_ids.index(product_id)
    sim_scores  = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    # Get top similar products excluding itself
    similar_indices = np.argsort(sim_scores)[::-1]
    results = []
    for i in similar_indices:
        pid = product_ids[i]
        if pid != product_id and sim_scores[i] > 0:
            results.append({
                "product_id": pid,
                "score":      round(float(sim_scores[i]), 4),
                "reason":     "Similar to products you viewed"
            })
        if len(results) >= n:
            break

    return results

def get_content_recommendations(purchased_ids,
                                  all_product_ids,
                                  n=10):
    """Recommend products based on what customer already bought."""
    if not purchased_ids:
        return []

    model = load_content_model()
    if not model:
        return []

    product_ids  = model['product_ids']
    tfidf_matrix = model['tfidf_matrix']

    # Get indices of purchased products
    purchased_indices = [
        product_ids.index(pid)
        for pid in purchased_ids
        if pid in product_ids
    ]

    if not purchased_indices:
        return []

    # Average feature vector of purchased products
    purchased_matrix = tfidf_matrix[purchased_indices]
    avg_vector       = purchased_matrix.mean(axis=0)

    # Compute similarity to all products
    sim_scores = cosine_similarity(avg_vector, tfidf_matrix).flatten()

    results = []
    for i in np.argsort(sim_scores)[::-1]:
        pid = product_ids[i]
        if pid not in purchased_ids and sim_scores[i] > 0:
            results.append({
                "product_id": pid,
                "score":      round(float(sim_scores[i]), 4),
                "reason":     "Based on your purchase history"
            })
        if len(results) >= n:
            break

    return results