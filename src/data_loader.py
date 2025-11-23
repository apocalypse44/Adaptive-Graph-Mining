"""
Data Loader Module
Handles loading and preprocessing of Amazon Review dataset
"""

import json
import gzip
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import pandas as pd


class AmazonDataLoader:
    """Load and preprocess Amazon review data"""
    
    def __init__(self, review_file: str = None, meta_file: str = None):
        """
        Initialize data loader
        
        Args:
            review_file: Path to review JSONL.gz file
            meta_file: Path to metadata JSONL.gz file
        """
        self.review_file = review_file
        self.meta_file = meta_file
        self.reviews = []
        self.metadata = {}
        
    def load_reviews(self, file_path: str, max_reviews: int = None) -> List[Dict]:
        """
        Load reviews from JSONL.gz file
        
        Args:
            file_path: Path to review file
            max_reviews: Maximum number of reviews to load (None for all)
            
        Returns:
            List of review dictionaries
        """
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
        
        self.reviews = reviews
        print(f"Total reviews loaded: {len(reviews)}")
        return reviews
    
    def load_metadata(self, file_path: str) -> Dict:
        """
        Load product metadata from JSONL.gz file
        
        Args:
            file_path: Path to metadata file
            
        Returns:
            Dictionary mapping parent_asin to metadata
        """
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
        """
        Extract co-purchase relationships from reviews
        Products are co-purchased if reviewed by the same user
        
        Returns:
            List of (product1, product2) tuples
        """
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
        """
        Extract co-view relationships from metadata
        Uses 'bought_together' field from metadata
        
        Returns:
            List of (product1, product2) tuples
        """
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
    
    def get_product_info(self, parent_asin: str) -> Dict:
        """Get metadata for a specific product"""
        return self.metadata.get(parent_asin, {})


