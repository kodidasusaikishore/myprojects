"""
prompt_dispatcher.py — Template Retrieval Engine

Loads prompt templates from the local JSON library and dispatches
them based on the classified intent category.
"""

import os
import json
from typing import Optional


class PromptDispatcher:
    """
    Manages the prompt template library and retrieves templates
    based on classified intent categories.
    """

    def __init__(self, library_path: str = None):
        if library_path is None:
            library_path = os.path.join(
                os.path.dirname(__file__), 'data', 'prompt_library.json'
            )
        self._library_path = library_path
        self._library = self._load_library()

    def _load_library(self) -> dict:
        """Load the prompt library from JSON."""
        with open(self._library_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_categories(self) -> list:
        """Return all available intent categories."""
        return list(self._library.keys())

    def get_category_info(self, category: str) -> Optional[dict]:
        """
        Get category description and template count.

        Returns:
            dict with 'description' and 'template_count', or None.
        """
        cat = self._library.get(category.upper())
        if cat is None:
            return None
        return {
            'description': cat.get('description', ''),
            'template_count': len(cat.get('templates', [])),
        }

    def get_templates(self, category: str, query: str = "") -> list:
        """
        Retrieve and rank prompt templates for a given category based on a query.

        Args:
            category: Intent category string (e.g., 'CODING', 'TESTING').
            query: User's original query for relevance ranking.

        Returns:
            List of template dicts, sorted by relevance score.
        """
        cat = self._library.get(category.upper())
        if cat is None:
            return []
        
        templates = cat.get('templates', [])
        
        if not query:
            return templates
            
        import re
        
        # Basic scoring algorithm
        ranked_results = []
        query_terms = [t for t in query.lower().split() if len(t) >= 2]
        
        if not query_terms:
            return [] # No meaningful terms to search with
            
        for t in templates:
            score = 0
            title_lower = t['title'].lower()
            desc_lower = t.get('description', '').lower()
            body_lower = t['template'].lower()
            
            for term in query_terms:
                # Use regex for word boundary matching to avoid substring leakage (e.g., 'hi' in 'think')
                pattern = rf'\b{re.escape(term)}\b'
                
                # Weighted matching with word boundaries
                if re.search(pattern, title_lower):
                    score += 15
                if re.search(pattern, desc_lower):
                    score += 5
                if re.search(pattern, body_lower):
                    score += 2
                    
            if score > 0:
                ranked_results.append((score, t))
        
        # Sort by score descending
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return sorted templates
        return [item[1] for item in ranked_results]

    def get_template_by_title(self, category: str, title: str) -> Optional[dict]:
        """Retrieve a specific template by category and title."""
        templates = self.get_templates(category)
        for t in templates:
            if t['title'].lower() == title.lower():
                return t
        return None

    def format_template(self, template_text: str, **kwargs) -> str:
        """
        Fill placeholders in a template string.

        Placeholders use {placeholder_name} format.
        Unfilled placeholders are left as-is.

        Args:
            template_text: The template string with {placeholders}.
            **kwargs: Key-value pairs to fill in.

        Returns:
            Template string with available placeholders filled.
        """
        result = template_text
        for key, value in kwargs.items():
            result = result.replace(f'{{{key}}}', str(value))
        return result

    def search_templates(self, query: str) -> list:
        """
        Search and rank templates across all categories.

        Args:
            query: Search keyword.

        Returns:
            List of (category, template) tuples, sorted by relevance.
        """
        if not query:
            return []
            
        import re
        query_terms = [t for t in query.lower().split() if len(t) >= 2]
        if not query_terms:
            return []
            
        results = []
        
        for category, data in self._library.items():
            for template in data.get('templates', []):
                score = 0
                title_lower = template['title'].lower()
                desc_lower = template.get('description', '').lower()
                body_lower = template['template'].strip().lower()
                
                for term in query_terms:
                    # Regex word boundaries to avoid partial matches (e.g., 'hi' in 'think')
                    pattern = rf'\b{re.escape(term)}\b'
                    if re.search(pattern, title_lower): score += 15
                    if re.search(pattern, desc_lower): score += 5
                    if re.search(pattern, body_lower): score += 1
                
                if score > 0:
                    results.append((score, category, template))
                    
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Return list of (category, template)
        return [(item[1], item[2]) for item in results]
