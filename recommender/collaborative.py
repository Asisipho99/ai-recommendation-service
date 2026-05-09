import pickle
import os
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

MODEL_PATH = "model/collaborative_model.pkl"

def train_collaborative_model(interactions):
    """Train SVD collaborative filtering model."""
    if len(interactions) < 5:
        print("Not enough interactions to train model.")
        return None

    df = pd.DataFrame(interactions)

    # Aggregate ratings per customer-product pair
    df = df.groupby(
        ['customer_id', 'product_id'],
        as_index=False
    )['rating'].max()

    reader  = Reader(rating_scale=(1, 5))
    data    = Dataset.load_from_df(
        df[['customer_id', 'product_id', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2,
                                          random_state=42)

    model = SVD(n_factors=50, n_epochs=20, random_state=42)
    model.fit(trainset)

    # Evaluate
    predictions = model.test(testset)
    rmse        = accuracy.rmse(predictions, verbose=False)
    print(f"Collaborative model RMSE: {rmse:.4f}")

    # Save model
    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "rmse": rmse}, f)

    return model, rmse

def load_collaborative_model():
    """Load saved collaborative filtering model."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data.get("model")

def get_collaborative_recommendations(customer_id,
                                       all_product_ids,
                                       purchased_ids,
                                       n=10):
    """Get top N product recommendations using collaborative filtering."""
    model = load_collaborative_model()
    if not model:
        return []

    # Only recommend products not already purchased
    candidates = [p for p in all_product_ids
                  if p not in purchased_ids]

    if not candidates:
        return []

    # Predict scores for all candidate products
    predictions = []
    for product_id in candidates:
        try:
            pred = model.predict(customer_id, product_id)
            predictions.append({
                "product_id": product_id,
                "score":      round(pred.est, 4),
                "reason":     "Customers like you also bought this"
            })
        except Exception:
            continue

    # Sort by predicted score descending
    predictions.sort(key=lambda x: x['score'], reverse=True)
    return predictions[:n]