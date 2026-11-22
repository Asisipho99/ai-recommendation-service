"""Interest-seeded recommendations - the cold-start tier between the hybrid
merge and global popularity.

It seeds, it does not filter. The tier only runs when the hybrid merge produced
nothing, so a brand new customer's stated interests bootstrap the panel and
then stop mattering the moment real interactions exist. For the same reason a
short interest result is topped up from global popularity rather than being
returned short: a customer who picks a thinly stocked category still sees a
full panel.

Deliberately free of any scikit-surprise import, so this module stays
importable on its own even where that compiled dependency is missing.
"""

from database.db import (get_customer_interests,
                         get_top_products,
                         get_top_products_by_categories)


def get_interest_recommendations(customer_id, n=5, exclude_ids=None):
    """Top products inside the customer's interest categories.

    Returns [] when the customer has no stored interests, which is the signal
    for the caller to fall through to the popularity tier.
    """
    exclude_ids = list(exclude_ids or [])

    categories = get_customer_interests(customer_id)
    if not categories:
        return []

    rows = get_top_products_by_categories(categories, n, exclude_ids=exclude_ids)

    items = [{
        "product_id": p['product_id'],
        "name":       p.get('name',     'Unknown'),
        "category":   p.get('category', 'Unknown'),
        "price":      float(p.get('price', 0)),
        "image_url":  p.get('image_url', ''),
        "score":      0.5,
        # The product's OWN category, verbatim from the catalogue row - not the
        # first interest and not a joined list. One product, one reason, one
        # category named.
        "reason":     f"Based on your interest in {p.get('category', 'Unknown')}"
    } for p in rows]

    if not items:
        return []

    # Top up from global popularity when the interest categories hold fewer
    # than n non-purchased active products.
    if len(items) < n:
        chosen_ids = [item['product_id'] for item in items]
        top_up = get_top_products(n - len(items),
                                  exclude_ids=exclude_ids + chosen_ids)
        items.extend({
            "product_id": p['product_id'],
            "name":       p.get('name',     'Unknown'),
            "category":   p.get('category', 'Unknown'),
            "price":      float(p.get('price', 0)),
            "image_url":  p.get('image_url', ''),
            "score":      0.5,
            "reason":     "Popular with our customers"
        } for p in top_up)

    return items
