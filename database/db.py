import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        database=os.getenv("MYSQL_DATABASE"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD")
    )


# The products, orders and order_items tables are created by Hibernate from the
# Spring entities. product_interactions is not: no service maps it, and there is
# no migration for it anywhere in the project. Yet every recommendation reads
# it and every logged view writes it, so on a fresh database the recommender
# fails with "table doesn't exist" and the API reports a cold start it cannot
# ever recover from. This service is the only user of the table, so it owns it.
#
# The UNIQUE key is not decoration: log_interaction() below treats
# (customer_id, product_id, interaction_type) as the identity of a row and
# hand-rolls a SELECT-then-UPDATE-or-INSERT upsert against it. Without the
# constraint, two concurrent writers both see no existing row and both insert,
# and the duplicate then skews the rating matrix. created_at needs a default
# because the INSERT never supplies it.
INTERACTIONS_DDL = """
    CREATE TABLE IF NOT EXISTS product_interactions (
        interaction_id   BIGINT       NOT NULL AUTO_INCREMENT,
        customer_id      BIGINT       NOT NULL,
        product_id       BIGINT       NOT NULL,
        interaction_type VARCHAR(32)  NOT NULL,
        rating           DOUBLE       NOT NULL DEFAULT 1,
        created_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (interaction_id),
        UNIQUE KEY uq_interaction (customer_id, product_id, interaction_type),
        KEY idx_interaction_customer (customer_id),
        KEY idx_interaction_product (product_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def ensure_schema():
    """Create product_interactions if it is absent.

    Safe to call repeatedly and from several gunicorn workers at once, since
    CREATE TABLE IF NOT EXISTS is idempotent.

    Deliberately does not raise. It runs at import time, and the service has
    useful routes that never touch MySQL - /health, and the whole dataset and
    ML pipeline, which read the filesystem. A database that is unreachable or
    not yet configured must not stop those from serving.

    Returns True when the table is present afterwards, False when the database
    could not be reached.
    """
    try:
        conn = get_connection()
    except mysql.connector.Error as exc:
        print(f"[schema] MySQL unreachable, skipping schema check: {exc}")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES LIKE 'product_interactions'")
        existed = cursor.fetchone() is not None
        cursor.execute(INTERACTIONS_DDL)
        conn.commit()
        cursor.close()
        if existed:
            print("[schema] product_interactions already present")
        else:
            print("[schema] created product_interactions")
        return True
    except mysql.connector.Error as exc:
        print(f"[schema] could not create product_interactions: {exc}")
        return False
    finally:
        conn.close()

def get_interactions():
    """Get all customer-product interactions for model training."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT customer_id, product_id, 
               interaction_type, rating,
               created_at
        FROM product_interactions
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_all_products():
    """Get all active products for content-based filtering.

    The products table uses a soft delete: the Java product-service sets
    deleted_at rather than removing the row, and filters on it. These raw
    queries have to apply the same filter or deleted products leak back into
    recommendations after an administrator has removed them.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT product_id, name, description, 
               category, price
        FROM products
        WHERE deleted_at IS NULL
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_customer_purchases(customer_id):
    """Get products a customer has already purchased."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT DISTINCT oi.product_id
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.customer_id = %s
        AND o.status != 'CANCELLED'
    """, (customer_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [r['product_id'] for r in rows]

def get_top_products(limit=5, exclude_ids=None):
    """Globally most-purchased active products, used as the cold-start fallback.

    `exclude_ids` lets the caller drop products the customer has already bought.
    Without it the fallback happily recommends something the customer just
    purchased, which is very visible on a small catalogue.
    """
    exclude_ids = list(exclude_ids or [])

    sql = """
        SELECT p.product_id, p.name, p.category,
               p.price, p.image_url,
               COUNT(oi.item_id) as purchase_count
        FROM products p
        LEFT JOIN order_items oi ON p.product_id = oi.product_id
        WHERE p.deleted_at IS NULL
    """
    params = []

    if exclude_ids:
        placeholders = ",".join(["%s"] * len(exclude_ids))
        sql += f" AND p.product_id NOT IN ({placeholders})"
        params.extend(exclude_ids)

    sql += """
        GROUP BY p.product_id
        ORDER BY purchase_count DESC
        LIMIT %s
    """
    params.append(limit)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, tuple(params))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# user-service OWNS the customer_interests table: it maps the rows with a JPA
# entity and Hibernate creates them (spring.jpa.hibernate.ddl-auto=update),
# exactly the way customers is managed. Unlike product_interactions above there
# is deliberately no DDL here and ensure_schema() must NOT be extended - this
# service only reads. If the table is absent because user-service has not been
# deployed yet, the mysql.connector.Error propagates and the recommendation
# route's existing except turns it into the degraded tier, which is the correct
# behaviour and is why the deployment order puts user-service first.
def get_customer_interests(customer_id):
    """Categories a customer picked as interests, ascending alphabetical."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT category
        FROM customer_interests
        WHERE customer_id = %s
        ORDER BY category ASC
    """, (customer_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [r['category'] for r in rows]

def get_top_products_by_categories(categories, limit=5, exclude_ids=None):
    """Most-purchased active products restricted to the given categories.

    The category-scoped twin of get_top_products, used to seed recommendations
    for a customer who has interests but no interaction history. `exclude_ids`
    drops products the customer already bought, as every other tier does.
    """
    categories = list(categories or [])
    if not categories:
        # No interests, no query - an empty list would emit `IN ()`, which is
        # not valid SQL.
        return []

    exclude_ids = list(exclude_ids or [])

    category_placeholders = ",".join(["%s"] * len(categories))
    sql = f"""
        SELECT p.product_id, p.name, p.category,
               p.price, p.image_url,
               COUNT(oi.item_id) as purchase_count
        FROM products p
        LEFT JOIN order_items oi ON p.product_id = oi.product_id
        WHERE p.deleted_at IS NULL
          AND p.category IN ({category_placeholders})
    """
    params = list(categories)

    if exclude_ids:
        placeholders = ",".join(["%s"] * len(exclude_ids))
        sql += f" AND p.product_id NOT IN ({placeholders})"
        params.extend(exclude_ids)

    sql += """
        GROUP BY p.product_id
        ORDER BY purchase_count DESC
        LIMIT %s
    """
    params.append(limit)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, tuple(params))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_product_details(product_ids):
    """Get full product details for a list of IDs."""
    if not product_ids:
        return []
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    format_strings = ','.join(['%s'] * len(product_ids))
    cursor.execute(f"""
        SELECT product_id, name, category, 
               price, image_url, description
        FROM products
        WHERE product_id IN ({format_strings})
          AND deleted_at IS NULL
    """, tuple(product_ids))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def log_interaction(customer_id, product_id, interaction_type):
    """Log a customer interaction event."""
    rating = 1.0 if interaction_type == 'view' else 5.0
    conn = get_connection()
    cursor = conn.cursor()

    # Check if interaction exists — update rating if so
    cursor.execute("""
        SELECT interaction_id FROM product_interactions
        WHERE customer_id = %s AND product_id = %s
        AND interaction_type = %s
    """, (customer_id, product_id, interaction_type))

    existing = cursor.fetchone()
    if existing:
        cursor.execute("""
            UPDATE product_interactions
            SET rating = %s, created_at = NOW()
            WHERE customer_id = %s
            AND product_id = %s
            AND interaction_type = %s
        """, (rating, customer_id, product_id, interaction_type))
    else:
        cursor.execute("""
            INSERT INTO product_interactions
            (customer_id, product_id, interaction_type, rating)
            VALUES (%s, %s, %s, %s)
        """, (customer_id, product_id, interaction_type, rating))

    conn.commit()
    cursor.close()
    conn.close()