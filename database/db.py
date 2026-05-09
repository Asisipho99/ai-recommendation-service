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
    """Get all products for content-based filtering."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT product_id, name, description, 
               category, price
        FROM products
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

def get_top_products(limit=5):
    """Get globally top-rated products as fallback."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT p.product_id, p.name, p.category,
               p.price, p.image_url,
               COUNT(oi.item_id) as purchase_count
        FROM products p
        LEFT JOIN order_items oi ON p.product_id = oi.product_id
        GROUP BY p.product_id
        ORDER BY purchase_count DESC
        LIMIT %s
    """, (limit,))
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