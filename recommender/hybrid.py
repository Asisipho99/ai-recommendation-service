from recommender.collaborative import get_collaborative_recommendations
from recommender.content_based  import get_content_recommendations
from recommender.interests      import get_interest_recommendations
from database.db                import (get_top_products,
                                        get_customer_interests,
                                        get_product_details,
                                        get_customer_purchases)

# Blend weights. Both inputs are put on a 0-1 scale first (see the two
# normalise_* helpers), so these weights are the real, effective split.
COLLABORATIVE_WEIGHT = 0.6
CONTENT_WEIGHT       = 0.4

# scikit-surprise predicts on the Reader rating scale used in
# collaborative.train_collaborative_model: (1, 5).
RATING_SCALE_MIN = 1.0
RATING_SCALE_MAX = 5.0


def normalise_collaborative_score(estimate):
    """Map an SVD rating estimate on the 1-5 scale onto 0-1.

    Fixed-scale min-max: (est - 1) / (5 - 1), clamped to [0, 1]. The scale is
    the rating scale the model was trained on, not the spread of one request's
    candidates, so a score means the same thing for every customer and request.
    """
    span = RATING_SCALE_MAX - RATING_SCALE_MIN
    return min(1.0, max(0.0, (float(estimate) - RATING_SCALE_MIN) / span))


def normalise_content_score(similarity):
    """Content scores are cosine similarities of non-negative TF-IDF vectors,
    already in [0, 1]. They are only clamped, never rescaled, so they stay
    comparable with the collaborative score on the same 0-1 scale."""
    return min(1.0, max(0.0, float(similarity)))


# A modest, flat nudge for products in a category the customer picked as an
# interest. It is added to the 0-1 blended score (so it is 5% of the scale),
# once per product however many interests match, and the result is capped at
# 1.0. It only re-scores products the collaborative and content strategies
# already surfaced: it never adds a candidate and never removes one, so it can
# reorder products whose blended scores sit within 0.05 of each other and
# nothing more. Interests are a nudge, not a filter.
INTEREST_BOOST = 0.05


def _interest_categories(customer_id, purchased_ids):
    """Lower-cased interest categories, or an empty set when no boost applies.

    Only customers with purchase history are boosted: a customer without any
    is served by the interest-seeded tier instead. A failed lookup means no
    boost, never a failed recommendation.
    """
    if not purchased_ids:
        return set()
    try:
        categories = get_customer_interests(customer_id)
    except Exception as exc:
        print(f"Interest lookup failed, ranking without a boost: {exc}")
        return set()
    return {str(c).strip().casefold() for c in categories or [] if c}


def get_hybrid_recommendations(customer_id,
                                all_product_ids,
                                n=5):
    """
    Hybrid recommendation combining collaborative
    and content-based filtering.

    Weights:
    - Collaborative filtering: 60%
    - Content-based filtering: 40%
    Each score is normalised to 0-1 before it is weighted, otherwise the 1-5
    collaborative rating estimate would outweigh the 0-1 content similarity
    by about an order of magnitude whatever the weights say.

    Returns (results, source) where source is "hybrid" when the merge produced
    something and "fallback" when it had to substitute popular products. The
    caller needs to know which, so the API can label the response honestly
    instead of always claiming a personalised result.
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
            "score":      (normalise_collaborative_score(rec['score'])
                           * COLLABORATIVE_WEIGHT),
            "reason":     rec['reason']
        }

    # Content-based — 40% weight
    for rec in content_recs:
        pid = rec['product_id']
        if pid in scores:
            scores[pid]['score']  += (normalise_content_score(rec['score'])
                                      * CONTENT_WEIGHT)
            scores[pid]['reason']  = "Recommended based on your activity"
        else:
            scores[pid] = {
                "product_id": pid,
                "score":      (normalise_content_score(rec['score'])
                               * CONTENT_WEIGHT),
                "reason":     rec['reason']
            }

    # Interest nudge for established customers: re-score the candidates the
    # strategies already produced, before the cut to n.
    product_map = {}
    interest_categories = (
        _interest_categories(customer_id, purchased_ids) if scores else set())
    if interest_categories:
        product_map = {
            p['product_id']: p
            for p in get_product_details(list(scores))
        }
        for pid, entry in scores.items():
            category = str(product_map.get(pid, {}).get('category') or '')
            if category.strip().casefold() in interest_categories:
                entry['score'] = min(1.0, entry['score'] + INTEREST_BOOST)
    # Sort by combined score
    sorted_recs = sorted(
        scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )[:n]

    if not sorted_recs:
        # Cold start: no interaction history to learn from yet. Try the
        # customer's stated interests before falling back to what is popular
        # with everybody.
        interest_recs = get_interest_recommendations(
            customer_id, n, exclude_ids=purchased_ids)
        if interest_recs:
            return interest_recs, "interest"
        return get_fallback_recommendations(n, exclude_ids=purchased_ids), "fallback"

    # Enrich with product details
    product_ids   = [r['product_id'] for r in sorted_recs
                     if r['product_id'] not in product_map]
    product_map.update({
        p['product_id']: p
        for p in get_product_details(product_ids)
    })

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

    return results, "hybrid"


def get_fallback_recommendations(n=5, exclude_ids=None):
    """
    Cold-start fallback — return top purchased
    products globally when no user data exists.

    `exclude_ids` keeps products the customer has already bought out of the
    list. The personalised path already excludes them; the fallback used not
    to, so a customer could be recommended the item they had just purchased.
    """
    top_products = get_top_products(n, exclude_ids=exclude_ids)
    return [{
        "product_id": p['product_id'],
        "name":       p.get('name',     'Unknown'),
        "category":   p.get('category', 'Unknown'),
        "price":      float(p.get('price', 0)),
        "image_url":  p.get('image_url', ''),
        "score":      0.5,
        "reason":     "Popular with our customers"
    } for p in top_products]