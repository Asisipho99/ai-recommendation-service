from recommender.collaborative import get_collaborative_recommendations
from recommender.content_based  import get_content_recommendations
from database.db                import (get_top_products,
                                        get_product_details,
                                        get_customer_purchases)

def get_hybrid_recommendations(customer_id,
                                all_product_ids,
                                n=5):
    """
    Hybrid recommendation combining collaborative
    and content-based filtering.

    Weights:
    - Collaborative filtering: 60%
    - Content-based filtering: 40%
    """
    purchased_ids = get_customer_purchases(customer_id)

    # Get recommendations from both approaches
    collab_recs  = get_collaborative_recommendations(
        customer_id, all_product_ids, purchased_ids, n=n*2)
    content_recs = get_content_recommendations(
        purchased_ids, all_product_ids, n=n*2)

    # Merge scores with weights
    scores = {}

    # Collaborative — 60% weight
    for rec in collab_recs:
        pid = rec['product_id']
        scores[pid] = {
            "product_id": pid,
            "score":      rec['score'] * 0.6,
            "reason":     rec['reason']
        }

    # Content-based — 40% weight
    for rec in content_recs:
        pid = rec['product_id']
        if pid in scores:
            scores[pid]['score']  += rec['score'] * 0.4
            scores[pid]['reason']  = "Recommended based on your activity"
        else:
            scores[pid] = {
                "product_id": pid,
                "score":      rec['score'] * 0.4,
                "reason":     rec['reason']
            }

    # Sort by combined score
    sorted_recs = sorted(
        scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )[:n]

    if not sorted_recs:
        return get_fallback_recommendations(n)

    # Enrich with product details
    product_ids   = [r['product_id'] for r in sorted_recs]
    product_map   = {
        p['product_id']: p
        for p in get_product_details(product_ids)
    }

    results = []
    for rec in sorted_recs:
        pid     = rec['product_id']
        product = product_map.get(pid, {})
        results.append({
            "product_id":  pid,
            "name":        product.get('name',     'Unknown'),
            "category":    product.get('category', 'Unknown'),
            "price":       float(product.get('price', 0)),
            "image_url":   product.get('image_url', ''),
            "score":       round(rec['score'], 4),
            "reason":      rec['reason']
        })

    return results


def get_fallback_recommendations(n=5):
    """
    Cold-start fallback — return top purchased
    products globally when no user data exists.
    """
    top_products = get_top_products(n)
    return [{
        "product_id": p['product_id'],
        "name":       p.get('name',     'Unknown'),
        "category":   p.get('category', 'Unknown'),
        "price":      float(p.get('price', 0)),
        "image_url":  p.get('image_url', ''),
        "score":      0.5,
        "reason":     "Popular with our customers"
    } for p in top_products]