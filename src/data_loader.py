import json
import gzip
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import pandas as pd
import random


class AmazonDataLoader:
    def __init__(self, review_file: str = None, meta_file: str = None):
        self.review_file = review_file
        self.meta_file = meta_file
        self.reviews = []
        self.metadata = {}
        
    def load_reviews(self, file_path: str, max_reviews: int = None, sample_ratio: float = 0.4) -> List[Dict]:
        reviews = []
        print(f"Loading reviews from {file_path}...")

        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_reviews and i >= max_reviews:
                    break
                try:
                    review = json.loads(line)
                    reviews.append(review)
                except json.JSONDecodeError:
                    continue

                if (i + 1) % 10000 == 0:
                    print(f"Loaded {i + 1} reviews...")

        print(f"Total reviews loaded before sampling: {len(reviews)}")

        # ---- NEW: Keep only 40% of reviews ---- #
        k = int(len(reviews) * sample_ratio)
        reviews = random.sample(reviews, k)

        print(f"Total reviews retained after {sample_ratio*100:.0f}% sampling: {len(reviews)}")

        self.reviews = reviews
        return reviews

    
    def load_metadata(self, file_path: str) -> Dict:
        metadata = {}
        print(f"Loading metadata from {file_path}...")
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    meta = json.loads(line)
                    parent_asin = meta.get('parent_asin')
                    if parent_asin:
                        metadata[parent_asin] = meta
                except json.JSONDecodeError:
                    continue
                    
                if (i + 1) % 10000 == 0:
                    print(f"Loaded {i + 1} metadata entries...")
        
        self.metadata = metadata
        print(f"Total metadata entries: {len(metadata)}")
        return metadata
    
    def extract_co_purchase_relationships(self) -> List[Tuple[str, str]]:
        user_products = defaultdict(set)
        
        print("Extracting co-purchase relationships...")
        for review in self.reviews:
            user_id = review.get('user_id')
            parent_asin = review.get('parent_asin')
            
            if user_id and parent_asin:
                user_products[user_id].add(parent_asin)
        
        # Generate co-purchase pairs
        co_purchases = []
        for user_id, products in user_products.items():
            products_list = list(products)
            # Create pairs of products reviewed by same user
            for i in range(len(products_list)):
                for j in range(i + 1, len(products_list)):
                    co_purchases.append((products_list[i], products_list[j]))
        
        print(f"Found {len(co_purchases)} co-purchase relationships")
        return co_purchases
    
    def extract_co_view_relationships(self) -> List[Tuple[str, str]]:
        co_views = []
        
        print("Extracting co-view relationships...")
        for parent_asin, meta in self.metadata.items():
            bought_together = meta.get('bought_together', [])
            if bought_together:
                for related_asin in bought_together:
                    if isinstance(related_asin, str):
                        co_views.append((parent_asin, related_asin))
        
        print(f"Found {len(co_views)} co-view relationships")
        return co_views
    
    def extract_co_review_relationships(self, time_window_days: int = 30) -> List[Tuple[str, str]]:
        """
        Build co-review edges based on timestamp proximity.
        Handles:
            - missing timestamps
            - invalid timestamps
            - millisecond timestamps
        """
        from datetime import datetime

        user_reviews = defaultdict(list)

        for r in self.reviews:
            user_id = r.get("user_id")
            asin = r.get("parent_asin")
            ts = r.get("timestamp")

            if not (user_id and asin and ts):
                continue

            # ---------- FIX: HANDLE INVALID TIMESTAMP VALUES ----------

            # 1. convert milliseconds to seconds if needed
            if ts > 10**12:      # definitely milliseconds
                ts = ts / 1000
            elif ts > 10**10:   # likely milliseconds
                ts = ts / 1000

            # 2. ignore negative or zero timestamps
            if ts <= 0:
                continue

            # 3. try converting to datetime
            try:
                dt = datetime.fromtimestamp(ts)
            except (OSError, OverflowError, ValueError):
                continue

            user_reviews[user_id].append((asin, dt))

        # ---------- CREATE EDGES ----------
        co_reviews = []

        for user, items in user_reviews.items():
            items.sort(key=lambda x: x[1])

            for i in range(len(items)):
                for j in range(i + 1, len(items)):

                    asin1, t1 = items[i]
                    asin2, t2 = items[j]

                    day_diff = abs((t2 - t1).days)

                    if day_diff <= time_window_days:
                        co_reviews.append((asin1, asin2))

        print(f"Found {len(co_reviews)} co-review relationships")
        return co_reviews


    
    def get_product_info(self, parent_asin: str) -> Dict:
        return self.metadata.get(parent_asin, {})


