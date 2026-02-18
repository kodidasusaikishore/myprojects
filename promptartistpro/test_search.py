import json
import re
import os

def test():
    lib_path = r'c:\Users\saiki\My_Projects\prompting_techniques_engine\data\prompt_library.json'
    with open(lib_path, 'r', encoding='utf-8') as f:
        lib = json.load(f)

    category = 'GENERAL'
    query = 'hi'
    
    templates = lib.get(category, {}).get('templates', [])
    
    query_terms = [t for t in query.lower().split() if len(t) >= 2]
    print(f"Query terms: {query_terms}")
    
    ranked_results = []
    for t in templates:
        score = 0
        title_lower = t['title'].lower()
        desc_lower = t.get('description', '').lower()
        body_lower = t['template'].lower()
        
        for term in query_terms:
            pattern = rf'\b{re.escape(term)}\b'
            if re.search(pattern, title_lower):
                score += 15
                print(f"Match in title: {t['title']}")
            if re.search(pattern, desc_lower):
                score += 5
                print(f"Match in desc: {t['title']}")
            if re.search(pattern, body_lower):
                score += 2
                print(f"Match in body: {t['title']}")
                
        if score > 0:
            ranked_results.append((score, t))
    
    ranked_results.sort(key=lambda x: x[0], reverse=True)
    results = [item[1]['title'] for item in ranked_results]
    print(f"Ranked results: {results}")

if __name__ == '__main__':
    test()
