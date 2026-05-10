from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os

from database.db import (
    get_interactions, get_all_products,
    get_product_details, log_interaction,
    get_top_products
)
from recommender.collaborative import (
    train_collaborative_model,
    load_collaborative_model
)
from recommender.content_based import (
    build_content_features,
    get_similar_products,
    load_content_model
)
from recommender.hybrid import (
    get_hybrid_recommendations,
    get_fallback_recommendations
)

load_dotenv()

app = Flask(__name__)

# Restrict CORS to known frontend origins
allowed_origins = os.getenv(
    'CORS_ALLOWED_ORIGINS',
    'http://localhost:4200,http://localhost:4201,'
    'https://shopping-admin-production.up.railway.app,'
    'https://shopping-customer-production.up.railway.app'
).split(',')

CORS(app, origins=[o.strip() for o in allowed_origins])

# ── Helper ────────────────────────────────────────────────────────
def success(data, status=200):
    return jsonify({"success": True,  "data": data}), status

def error(message, status=400):
    return jsonify({"success": False, "error": message}), status


# ── Health Check ──────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return success({
        "status":  "running",
        "service": "AI Recommendation Service",
        "version": "1.0.0"
    })


# ── Get Recommendations for a Customer ───────────────────────────
@app.route('/api/recommendations/<int:customer_id>', methods=['GET'])
def get_recommendations(customer_id):
    try:
        n = int(request.args.get('n', 5))

        # Get all product IDs
        products       = get_all_products()
        all_product_ids = [p['product_id'] for p in products]

        if not all_product_ids:
            return success([])

        # Try hybrid recommendations
        recommendations = get_hybrid_recommendations(
            customer_id, all_product_ids, n=n)

        # Fallback if no recommendations
        if not recommendations:
            recommendations = get_fallback_recommendations(n)

        return success({
            "customer_id":     customer_id,
            "recommendations": recommendations,
            "count":           len(recommendations),
            "type":            "hybrid"
        })

    except Exception as e:
        print(f"Recommendation error: {e}")
        # Always return something — fallback on any error
        try:
            fallback = get_fallback_recommendations(5)
            return success({
                "customer_id":     customer_id,
                "recommendations": fallback,
                "count":           len(fallback),
                "type":            "fallback"
            })
        except:
            return error(str(e))


# ── Get Similar Products ──────────────────────────────────────────
@app.route('/api/recommendations/similar/<int:product_id>',
           methods=['GET'])
def get_similar(product_id):
    try:
        n       = int(request.args.get('n', 5))
        similar = get_similar_products(product_id, n=n)

        if not similar:
            # Build content model if not exists
            products = get_all_products()
            build_content_features(products)
            similar  = get_similar_products(product_id, n=n)

        product_ids  = [r['product_id'] for r in similar]
        product_map  = {
            p['product_id']: p
            for p in get_product_details(product_ids)
        }

        results = []
        for rec in similar:
            pid     = rec['product_id']
            product = product_map.get(pid, {})
            results.append({
                "product_id": pid,
                "name":       product.get('name',     'Unknown'),
                "category":   product.get('category', 'Unknown'),
                "price":      float(product.get('price', 0)),
                "image_url":  product.get('image_url', ''),
                "score":      rec['score'],
                "reason":     "Customers also viewed this"
            })

        return success({
            "product_id": product_id,
            "similar":    results,
            "count":      len(results)
        })

    except Exception as e:
        return error(str(e))


# ── Log Customer Interaction ──────────────────────────────────────
@app.route('/api/recommendations/interactions', methods=['POST'])
def log_interaction_event():
    try:
        data             = request.get_json()
        customer_id      = data.get('customerId')
        product_id       = data.get('productId')
        interaction_type = data.get('interactionType', 'view')

        if not customer_id or not product_id:
            return error("customerId and productId are required")

        if interaction_type not in ['view', 'purchase']:
            return error("interactionType must be 'view' or 'purchase'")

        log_interaction(customer_id, product_id, interaction_type)

        return success({
            "message":         "Interaction logged successfully",
            "customer_id":     customer_id,
            "product_id":      product_id,
            "interaction_type": interaction_type
        })

    except Exception as e:
        return error(str(e))


# ── Train / Retrain Models ────────────────────────────────────────
@app.route('/api/recommendations/retrain', methods=['POST'])
def retrain_models():
    try:
        print("Starting model retraining...")

        # Train collaborative filtering
        interactions = get_interactions()
        collab_result = None
        if len(interactions) >= 5:
            model, rmse   = train_collaborative_model(interactions)
            collab_result = {
                "status": "trained",
                "rmse":   round(rmse, 4),
                "interactions_used": len(interactions)
            }
        else:
            collab_result = {
                "status":  "skipped",
                "reason":  f"Only {len(interactions)} interactions "
                           f"— need at least 5"
            }

        # Train content-based
        products = get_all_products()
        build_content_features(products)
        content_result = {
            "status":        "trained",
            "products_used": len(products)
        }

        return success({
            "message":      "Models retrained successfully",
            "collaborative": collab_result,
            "content_based": content_result
        })

    except Exception as e:
        return error(str(e), 500)


# ── Model Status ──────────────────────────────────────────────────
@app.route('/api/recommendations/status', methods=['GET'])
def model_status():
    try:
        collab_model  = load_collaborative_model()
        content_model = load_content_model()

        interactions = get_interactions()
        products     = get_all_products()

        return success({
            "collaborative_model": {
                "loaded":   collab_model is not None,
                "status":   "ready" if collab_model else "not trained"
            },
            "content_model": {
                "loaded":   content_model is not None,
                "status":   "ready" if content_model else "not trained"
            },
            "data": {
                "total_interactions": len(interactions),
                "total_products":     len(products)
            }
        })

    except Exception as e:
        return error(str(e))


# ── Popular Products ──────────────────────────────────────────────
@app.route('/api/recommendations/popular', methods=['GET'])
def popular_products():
    try:
        n        = int(request.args.get('n', 5))
        products = get_top_products(n)
        return success({
            "popular": products,
            "count":   len(products)
        })
    except Exception as e:
        return error(str(e))


# ── Run ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"AI Recommendation Service starting on port {port}")

    # On startup, only retrain if no saved model exists.
    # If models are already saved (from a previous run), load them directly
    # to avoid losing trained state on container restart.
    try:
        print("Checking for saved models...")
        collab_exists  = os.path.exists("model/collaborative_model.pkl")
        content_exists = os.path.exists("model/content_model.pkl")

        if content_exists:
            print("Content model found — loading from disk, skipping retraining")
        else:
            print("No content model found — training from scratch...")
            products = get_all_products()
            if products:
                build_content_features(products)
                print(f"Content model trained on {len(products)} products")
            else:
                print("No products available — content model skipped")

        if collab_exists:
            print("Collaborative model found — loading from disk, skipping retraining")
        else:
            print("No collaborative model found — training from scratch...")
            interactions = get_interactions()
            if len(interactions) >= 5:
                train_collaborative_model(interactions)
                print(f"Collaborative model trained on {len(interactions)} interactions")
            else:
                print(f"Only {len(interactions)} interactions — "
                      f"collaborative model skipped (need at least 5)")

    except Exception as e:
        print(f"Startup model check failed: {e}")

    app.run(host='0.0.0.0', port=port, debug=False)