"""
Text Encoder for outfit prompts.
Uses TF-IDF + category keywords matching for prototype.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Fashion-related keywords mapped to categories
CATEGORY_KEYWORDS = {
    'tops': ['shirt', 'blouse', 'top', 'tee', 't-shirt', 'sweater', 'hoodie', 'cardigan', 'jacket', 'coat'],
    'bottoms': ['pants', 'jeans', 'shorts', 'skirt', 'trousers', 'leggings'],
    'dresses': ['dress', 'gown', 'romper', 'jumpsuit', 'overall'],
    'footwear': ['shoes', 'sneakers', 'boots', 'sandals', 'heels', 'flats', 'loafers', 'slippers'],
    'accessories': ['hat', 'cap', 'beanie', 'scarf', 'belt', 'watch', 'sunglasses', 'glasses', 'jewelry', 'necklace', 'bracelet', 'earring', 'ring', 'bag', 'purse', 'backpack'],
    'outerwear': ['jacket', 'coat', 'blazer', 'vest', 'parka', 'windbreaker'],
}

# Mood/style keywords
STYLE_KEYWORDS = {
    'casual': ['casual', 'relaxed', 'everyday', 'cozy', 'comfortable', 'laid-back'],
    'formal': ['formal', 'elegant', 'dressy', 'professional', 'business', 'sophisticated'],
    'sporty': ['sporty', 'athletic', 'fitness', 'active', 'workout', 'gym'],
    'summer': ['summer', 'beach', 'sunny', 'tropical', 'light', 'breezy', 'warm'],
    'winter': ['winter', 'cold', 'warm', 'cozy', 'snow', 'frost'],
    'spring': ['spring', 'floral', 'fresh', 'light', 'pastel'],
    'fall': ['fall', 'autumn', 'cozy', 'warm', 'layered'],
    'boho': ['boho', 'bohemian', 'hippie', 'artistic', 'free-spirited'],
    'vintage': ['vintage', 'retro', 'classic', 'old-fashioned', 'nostalgic'],
    'modern': ['modern', 'contemporary', 'sleek', 'minimalist', 'clean'],
    'edgy': ['edgy', 'punk', 'goth', 'dark', 'bold'],
    'romantic': ['romantic', 'feminine', 'soft', 'delicate', 'sweet'],
    'streetwear': ['streetwear', 'urban', 'hip-hop', 'skate', 'graffiti'],
    'minimalist': ['minimalist', 'simple', 'basic', 'clean', 'essential'],
}

# Color keywords
COLOR_KEYWORDS = {
    'red': ['red', 'crimson', 'scarlet', 'ruby', 'maroon', 'burgundy'],
    'blue': ['blue', 'navy', 'royal', 'cobalt', 'azure', 'indigo', 'denim'],
    'green': ['green', 'emerald', 'sage', 'olive', 'forest', 'mint'],
    'yellow': ['yellow', 'gold', 'mustard', 'lemon', 'honey'],
    'orange': ['orange', 'peach', 'coral', 'tangerine', 'apricot'],
    'pink': ['pink', 'rose', 'blush', 'fuchsia', 'magenta'],
    'purple': ['purple', 'violet', 'lavender', 'plum', 'iris'],
    'black': ['black', 'noir', 'obsidian', 'onyx'],
    'white': ['white', 'ivory', 'cream', 'off-white', 'pearl'],
    'gray': ['gray', 'grey', 'silver', 'charcoal', 'slate'],
    'brown': ['brown', 'tan', 'caramel', 'chocolate', 'coffee', 'bronze'],
    'beige': ['beige', 'nude', 'champagne', 'fawn', 'sand'],
}


class TextEncoder:
    """
    Encodes natural language prompts into embeddings for outfit generation.
    Prototype version uses keyword matching + TF-IDF.
    """

    def __init__(self, categories):
        self.categories = categories
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

        # Build corpus from all keywords
        corpus = []
        corpus.extend([' '.join(v) for v in CATEGORY_KEYWORDS.values()])
        corpus.extend([' '.join(v) for v in STYLE_KEYWORDS.values()])
        corpus.extend([' '.join(v) for v in COLOR_KEYWORDS.values()])

        # Add category names
        category_names = [c.lower().replace('_', ' ') for c in categories]
        corpus.extend(category_names)

        self.vectorizer.fit(corpus)

    def encode(self, prompt):
        """
        Encode a text prompt into a feature vector.

        Args:
            prompt: Natural language prompt string

        Returns:
            dict with:
                - embedding: TF-IDF vector (100-dim)
                - keywords: Dict of extracted keywords (category, style, color)
                - original: Original prompt
        """
        prompt_lower = prompt.lower()

        # Extract keywords
        keywords = {
            'categories': [],
            'styles': [],
            'colors': [],
            'moods': []
        }

        # Check each keyword category
        for keyword_list, keyword_type in [
            (CATEGORY_KEYWORDS, 'categories'),
            (STYLE_KEYWORDS, 'styles'),
            (COLOR_KEYWORDS, 'colors')
        ]:
            for category, words in keyword_list.items():
                for word in words:
                    if word in prompt_lower:
                        keywords[keyword_type].append(category)
                        break

        # Mood extraction (use style keywords for mood)
        for style, words in STYLE_KEYWORDS.items():
            for word in words:
                if word in prompt_lower:
                    keywords['moods'].append(style)
                    break

        # TF-IDF embedding
        embedding = self.vectorizer.transform([prompt]).toarray()[0]

        return {
            'embedding': embedding,
            'keywords': keywords,
            'original': prompt
        }

    def match_categories(self, keywords, top_k=5):
        """
        Match extracted keywords to dataset categories.

        Args:
            keywords: Dict with 'categories' list
            top_k: Number of categories to return

        Returns:
            List of matched category IDs
        """
        matched = []
        for kw in keywords.get('categories', []):
            for cat_id, cat_name in enumerate(self.categories):
                cat_lower = cat_name.lower()
                if kw in cat_lower or any(w in cat_lower for w in CATEGORY_KEYWORDS.get(kw, [])):
                    matched.append(cat_id)
        return list(set(matched))[:top_k]


def encode_outfit_items(items, categories):
    """
    Encode outfit items for comparison with text prompt.

    Args:
        items: List of item dicts with 'category_id', 'name', 'description'
        categories: List of category names

    Returns:
        List of encoded item vectors
    """
    encoder = TextEncoder(categories)
    vectors = []

    for item in items:
        text = f"{item.get('name', '')} {item.get('description', '')} {categories[item['category_id']]}"
        vec = encoder.vectorizer.transform([text]).toarray()[0]
        vectors.append(vec)

    return vectors
