from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import traceback
from datetime import datetime, timezone

from database.db import (
    get_interactions, get_all_products,
    get_product_details, log_interaction,
    get_top_products, ensure_schema
)

# The recommender modules are imported lazily inside their handlers. They pull
# in scikit-surprise, which is a compiled dependency that is not always
# available outside the container. Deferring the import means the dataset and ML
# routes - and the test suite - keep working without it, and startup is faster.

load_dotenv()

app = Flask(__name__)

# Restrict CORS to known frontend origins.
#
# `or` rather than a getenv default on purpose. os.getenv returns '' for a
# variable that is set but empty, which is not None, so the default would be
# skipped and origins would become [''] - an allowlist matching nothing. That
# rejects every origin, including localhost, while the service still answers
# preflights with 200 and no Access-Control-Allow-Origin header. The browser
# then blocks the real request and the only trace is a flood of OPTIONS with
# no matching GET, which is almost impossible to read as a configuration
# problem. Empty entries are filtered for the same reason.
DEFAULT_ORIGINS = (
    'http://localhost:4200,http://localhost:4201,'
    'https://shopping-admin-production.up.railway.app,'
    'https://shopping-customer-production.up.railway.app'
)

allowed_origins = [
    o.strip()
    for o in (os.getenv('CORS_ALLOWED_ORIGINS') or DEFAULT_ORIGINS).split(',')
    if o.strip()
] or [o.strip() for o in DEFAULT_ORIGINS.split(',')]

CORS(app, origins=allowed_origins)
print(f"[cors] allowing {len(allowed_origins)} origin(s): "
      f"{', '.join(allowed_origins)}")

# Run at import, not under `if __name__ == '__main__'`. The container starts
# gunicorn, which imports this module rather than executing it, so anything in
# the __main__ block never runs in production - which is exactly how the
# missing product_interactions table went unnoticed. ensure_schema() swallows
# connection errors so an unconfigured database still leaves /health and the
# dataset routes serving.
ensure_schema()


def warm_models():
    """Train any recommender model that is missing from disk.

    The container filesystem is ephemeral. Every deployment starts a fresh
    container with an empty model/ directory, while the interactions that
    train the models live in MySQL and survive. The pickles therefore vanish
    on each deploy, and because the retraining logic lived in the __main__
    block - which gunicorn never executes - a freshly deployed instance
    served popularity-only results until somebody remembered to POST
    /api/recommendations/retrain by hand. Nothing in the response said so:
    the degraded state was reported as an ordinary cold start.

    Only trains what is absent, so a restart that still has its models is
    cheap. Every failure is swallowed: scikit-surprise is a compiled
    dependency that is not always importable, the database may not be
    configured yet, and neither should stop /health or the filesystem-backed
    dataset and ML routes from serving.
    """
    try:
        products = get_all_products()
    except Exception as exc:                                  # noqa: BLE001
        print(f"[warm] database unreachable, skipping model warm-up: {exc}")
        return

    if not os.path.exists("model/content_model.pkl"):
        try:
            from recommender.content_based import build_content_features
            if products:
                build_content_features(products)
                print(f"[warm] content model trained on {len(products)} "
                      f"products")
            else:
                print("[warm] no products, content model skipped")
        except Exception as exc:                              # noqa: BLE001
            print(f"[warm] content model training failed: "
                  f"{type(exc).__name__}: {exc}")
    else:
        print("[warm] content model already on disk")

    if not os.path.exists("model/collaborative_model.pkl"):
        try:
            from recommender.collaborative import train_collaborative_model
            interactions = get_interactions()
            if len(interactions) >= 5:
                train_collaborative_model(interactions)
                print(f"[warm] collaborative model trained on "
                      f"{len(interactions)} interactions")
            else:
                print(f"[warm] only {len(interactions)} interactions, "
                      f"collaborative model needs 5")
        except Exception as exc:                              # noqa: BLE001
            print(f"[warm] collaborative model training failed: "
                  f"{type(exc).__name__}: {exc}")
    else:
        print("[warm] collaborative model already on disk")


warm_models()

# How many raw rows the data pipeline will read. The full dataset is about a
# million rows; capping it keeps a small container within its memory budget.
DATA_ROW_LIMIT = int(os.getenv('DATA_ROW_LIMIT', '0')) or None


# ── Helper ────────────────────────────────────────────────────────────
def success(data, status=200):
    return jsonify({"success": True,  "data": data}), status

def error(message, status=400):
    return jsonify({"success": False, "error": message}), status


# ── Health Check ──────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return success({
        "status":  "running",
        "service": "AI Recommendation & Analytics Service",
        "version": "2.0.0"
    })


# ══════════════════════════════════════════════════════════════════════
#  DATASET, EDA AND DATA PREPARATION
#
#  These routes operate on an external dataset (Online Retail II, UCI /
#  Kaggle) rather than on data this platform generated about itself.
#  Heavy imports are deliberately inside the handlers so the service
#  starts quickly and does not pay for pandas unless a route needs it.
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/data/source', methods=['GET'])
def data_source():
    """What dataset is on disk, and its provenance."""
    from dataset import loader

    info = loader.source_info()
    info["dataset"] = {
        "name": "Online Retail II",
        "origin": "UCI Machine Learning Repository (also mirrored on Kaggle)",
        "description": "Real UK online-retail transactions, Dec 2009 - Dec 2011.",
        "why": "An external, pre-existing dataset with genuine data-quality "
               "problems, rather than data generated by this application.",
        "row_limit_applied": DATA_ROW_LIMIT,
    }
    return success(info)


@app.route('/api/data/eda', methods=['GET'])
def data_eda():
    """Profile the raw dataset: shape, missing values, distributions, problems."""
    from dataset import eda, loader

    try:
        frame = loader.load_raw(limit=DATA_ROW_LIMIT)
    except loader.DatasetNotFound as exc:
        return error(str(exc), 404)

    return success(eda.full_report(frame))


@app.route('/api/data/distribution/<column>', methods=['GET'])
def data_distribution(column):
    """Histogram of one numeric column, for charting."""
    from dataset import eda, loader

    try:
        frame = loader.load_raw(limit=DATA_ROW_LIMIT)
    except loader.DatasetNotFound as exc:
        return error(str(exc), 404)

    bins = int(request.args.get('bins', 20))
    return success(eda.distribution(frame, column, bins=bins))


@app.route('/api/data/prepare', methods=['POST'])
def data_prepare():
    """Clean the dataset and build the customer feature table.

    Returns the full audit trail: what was removed at each step and why.
    """
    from dataset import loader, prepare

    try:
        report = prepare.run_pipeline(limit=DATA_ROW_LIMIT)
    except loader.DatasetNotFound as exc:
        return error(str(exc), 404)
    except Exception as exc:  # noqa: BLE001
        return error(f"preparation failed: {exc}", 500)

    return success(report)


# ══════════════════════════════════════════════════════════════════════
#  MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/ml/train', methods=['POST'])
def ml_train():
    """Train all four algorithms and produce the comparison."""
    from ml import registry

    try:
        return success(registry.train_all())
    except FileNotFoundError as exc:
        return error(str(exc), 409)
    except ValueError as exc:
        return error(str(exc), 422)
    except Exception as exc:  # noqa: BLE001
        return error(f"training failed: {exc}", 500)


@app.route('/api/ml/comparison', methods=['GET'])
def ml_comparison():
    """The algorithm comparison from the last training run."""
    from ml import registry

    report = registry.load_report()
    if report is None:
        return error("no training report yet; POST /api/ml/train first", 404)
    return success(report)


@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Which models are trained and ready."""
    from ml import registry

    return success(registry.status())


@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    """Score one customer against a chosen model."""
    from ml import registry

    payload = request.get_json(silent=True) or {}
    model_name = payload.get('model', 'Random Forest')
    features = payload.get('features')

    if not isinstance(features, dict) or not features:
        return error("'features' must be an object of feature name to value")

    try:
        return success(registry.predict(model_name, features))
    except FileNotFoundError as exc:
        return error(str(exc), 409)
    except KeyError as exc:
        return error(str(exc).strip("\"'"), 404)
    except (ValueError, TypeError) as exc:
        return error(str(exc), 422)


# ══════════════════════════════════════════════════════════════════════
#  RECOMMENDER (operates on live platform data)
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/recommendations/<int:customer_id>', methods=['GET'])
def get_recommendations(customer_id):
    from recommender.hybrid import (
        get_fallback_recommendations,
        get_hybrid_recommendations,
    )

    # Parsed before the try so the exception handler can still honour it. A
    # bad ?n= is the caller's mistake, not a server failure, so it is rejected
    # rather than silently defaulted.
    try:
        n = int(request.args.get('n', 5))
    except (TypeError, ValueError):
        return error("'n' must be an integer")
    n = max(1, min(n, 100))

    try:
        products       = get_all_products()
        all_product_ids = [p['product_id'] for p in products]

        if not all_product_ids:
            return success({
                "customer_id":     customer_id,
                "recommendations": [],
                "count":           0,
                "type":            "empty",
                "generated_at":    _now()
            })

        # `source` tells us whether this was a real personalised result or the
        # popular-products substitute, so the response can say which.
        recommendations, source = get_hybrid_recommendations(
            customer_id, all_product_ids, n=n)

        return success({
            "customer_id":     customer_id,
            "recommendations": recommendations,
            "count":           len(recommendations),
            "type":            source,
            "generated_at":    _now()
        })

    except Exception as e:
        # Always return something usable — a failure here must not break the
        # page that displays it. But do not pretend a failure is a cold start:
        # both used to answer with an identical {"type": "fallback"}, so a
        # genuine error was indistinguishable from a customer who simply has
        # no history, and the only clue was this log line. `degraded` makes the
        # difference visible to a caller. It is an additive field, so the
        # frontends' 'hybrid' | 'fallback' union still holds.
        print(f"Recommendation error for customer {customer_id}: "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            # Honour the requested n. Hardcoding 5 here meant a degraded
            # response was silently capped, which is what made this path
            # identifiable from outside but was never intended.
            fallback = get_fallback_recommendations(n)
            return success({
                "customer_id":     customer_id,
                "recommendations": fallback,
                "count":           len(fallback),
                "type":            "fallback",
                "degraded":        True,
                "generated_at":    _now()
            })
        except Exception:
            return error(str(e))


@app.route('/api/recommendations/similar/<int:product_id>',
           methods=['GET'])
def get_similar(product_id):
    from recommender.content_based import (
        build_content_features,
        get_similar_products,
    )

    try:
        n       = int(request.args.get('n', 5))
        similar = get_similar_products(product_id, n=n)

        if not similar:
            # Build the content model if it does not exist yet
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


@app.route('/api/recommendations/retrain', methods=['POST'])
def retrain_models():
    from recommender.collaborative import train_collaborative_model
    from recommender.content_based import build_content_features

    try:
        print("Starting model retraining...")

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


@app.route('/api/recommendations/status', methods=['GET'])
def model_status():
    from recommender.collaborative import load_collaborative_model
    from recommender.content_based import load_content_model

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


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"AI Recommendation Service starting on port {port}")
    try:
        from recommender.collaborative import train_collaborative_model
        from recommender.content_based import build_content_features

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
