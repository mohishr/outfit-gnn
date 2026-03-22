"""
MongoDB Data Loader for Outfit GNN.
Fetches data from MongoDB at localhost:27021, db=fashion_recommendation_db, collection=Clothing_Items.
Uses batch queries and caching for efficiency.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from pymongo import MongoClient


class MongoOutfitDataset:
    """
    Dataset loaded from MongoDB Clothing_Items collection.

    Document structure:
    {
        "_id": ObjectId,           # Used as item_id (format: {outfit_id}_{item_index})
        "user_id": null,
        "category_id": 17,
        "category": "Blouses",
        "description": "Blouses",
        "image_blob": {"$binary": {"base64": "..."}},
        "image_embedding": [],
        "text_embedding": null
    }

    Item ID format: {outfit_id}_{item_index} where item_index is derived from document position
    """

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27021",
        db_name: str = "fashion_recommendation_db",
        collection_name: str = "Clothing_Items",
        cache_dir: str = "data"
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Connect to MongoDB
        self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Load cached category info
        self._load_category_cache()

        # Statistics
        self.num_train = 0
        self.num_test = 0

        # Cache for items
        self._item_cache = {}

    def _load_category_cache(self):
        """Load or build category cache from MongoDB data."""
        cache_file = self.cache_dir / "mongo_categories.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                self.category_names = {int(k): v for k, v in cache['category_names'].items()}
                self.categories = cache['categories']
                self.num_categories = len(self.categories)
                self.cid2rcid = cache['cid2rcid']
                print(f"Loaded category cache with {self.num_categories} categories")
        else:
            self._build_category_cache(cache_file)

    def _build_category_cache(self, cache_file: Path):
        """Build category cache from MongoDB data."""
        print("Building category cache from MongoDB...")
        # Get distinct categories from MongoDB
        pipeline = [
            {"$group": {"_id": "$category_id", "name": {"$first": "$category"}}},
            {"$sort": {"_id": 1}}
        ]
        categories_data = list(self.collection.aggregate(pipeline))

        self.category_names = {}
        self.categories = []
        self.cid2rcid = {}

        for idx, doc in enumerate(sorted(categories_data, key=lambda x: x['_id'])):
            cat_id = doc['_id']
            name = doc['name']
            self.category_names[cat_id] = name
            self.categories.append(name)
            self.cid2rcid[str(cat_id)] = idx

        self.num_categories = len(self.categories)

        # Save cache
        cache = {
            'category_names': {str(k): v for k, v in self.category_names.items()},
            'categories': self.categories,
            'cid2rcid': self.cid2rcid
        }
        with open(cache_file, 'w') as f:
            json.dump(cache, f)

        print(f"Built category cache with {self.num_categories} categories")

    def load_items(self, limit: Optional[int] = None, skip: int = 0) -> List[Dict]:
        """
        Load all clothing items from MongoDB.

        Returns:
            List of item dicts with category info and image data.
        """
        query = {}
        if limit:
            cursor = self.collection.find(query, {
                '_id': 1, 'category_id': 1, 'category': 1, 'description': 1
            }).skip(skip).limit(limit)
        else:
            cursor = self.collection.find(query, {
                '_id': 1, 'category_id': 1, 'category': 1, 'description': 1
            }).skip(skip)

        items = []
        for doc in cursor:
            item = {
                '_id': str(doc.get('_id', '')),
                'category_id': doc.get('category_id'),
                'category': doc.get('category', 'Unknown'),
                'description': doc.get('description', ''),
            }
            items.append(item)

        return items

    def get_items_by_category(self, category_id: int, limit: int = 100) -> List[Dict]:
        """Get items for a specific category (with caching)."""
        cache_key = f"cat_{category_id}_{limit}"
        if cache_key in self._item_cache:
            return self._item_cache[cache_key]

        cursor = self.collection.find(
            {'category_id': category_id},
            {'_id': 1, 'category_id': 1, 'category': 1, 'description': 1}
        ).limit(limit)

        items = []
        for doc in cursor:
            item = {
                '_id': str(doc.get('_id', '')),
                'category_id': doc.get('category_id'),
                'category': doc.get('category', 'Unknown'),
                'description': doc.get('description', ''),
            }
            items.append(item)

        self._item_cache[cache_key] = items
        return items

    def get_reduced_category_id(self, category_id: int) -> Optional[int]:
        """Map original category ID to reduced consecutive ID."""
        return self.cid2rcid.get(str(category_id))

    def get_category_name(self, category_id: int) -> str:
        """Get category name from ID."""
        return self.category_names.get(category_id, 'Unknown')

    def build_item_pool(self, limit_per_category: int = 100) -> List[Dict]:
        """
        Build an item pool for outfit generation.
        Returns items with their category embeddings and visual features placeholders.
        """
        print(f"Building item pool with limit {limit_per_category} per category...")
        item_pool = []

        cat_ids = list(self.category_names.keys())
        for i, cat_id in enumerate(cat_ids):
            if i % 20 == 0:
                print(f"  Processing category {i+1}/{len(cat_ids)}...", flush=True)

            items = self.get_items_by_category(cat_id, limit=limit_per_category)
            reduced_cat_id = self.get_reduced_category_id(cat_id)

            for item in items:
                # Parse item_id from MongoDB _id (format: outfitid_itemindex)
                item_id = item['_id']
                parts = item_id.rsplit('_', 1)
                outfit_id = parts[0] if len(parts) > 0 else item_id
                item_index = parts[1] if len(parts) > 1 else '0'

                pool_item = {
                    'item_id': item_id,
                    'outfit_id': outfit_id,
                    'item_index': item_index,
                    'category_id': cat_id,
                    'reduced_cat_id': reduced_cat_id,
                    'category': item.get('category', 'Unknown'),
                    'description': item.get('description', ''),
                    'image_base64': '',
                    'visual_features': None  # Will use category embedding as placeholder
                }
                item_pool.append(pool_item)

        print(f"Built item pool with {len(item_pool):,} items")
        return item_pool

    def get_item_image(self, item_id: str) -> Optional[str]:
        """Get base64 image for an item."""
        cache_key = f"img_{item_id}"
        if cache_key in self._item_cache:
            return self._item_cache[cache_key]

        from bson import ObjectId
        try:
            doc = self.collection.find_one({'_id': ObjectId(item_id)})
        except:
            # Try string lookup
            doc = self.collection.find_one({'_id': item_id})

        if doc:
            image_blob = doc.get('image_blob')
            if image_blob and isinstance(image_blob, dict):
                base64_data = image_blob.get('$binary', {}).get('base64', '')
                self._item_cache[cache_key] = base64_data
                return base64_data

        self._item_cache[cache_key] = None
        return None

    def get_category_distribution(self) -> Dict[str, int]:
        """Get count of items per category."""
        pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        result = {}
        for doc in self.collection.aggregate(pipeline):
            result[doc['_id']] = doc['count']
        return result

    def summary(self):
        """Print dataset summary."""
        print("=" * 50)
        print("MongoOutfitDataset Summary")
        print("=" * 50)
        print(f"Total categories: {self.num_categories}")
        print(f"Categories: {self.categories[:10]}...")
        print(f"Cache entries: {len(self._item_cache)}")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def load_from_mongo(**kwargs) -> MongoOutfitDataset:
    """Convenience function to create MongoOutfitDataset."""
    return MongoOutfitDataset(**kwargs)


if __name__ == "__main__":
    dataset = MongoOutfitDataset()
    dataset.summary()
    pool = dataset.build_item_pool(limit_per_category=50)
    print(f"Pool size: {len(pool)}")
    dataset.close()
