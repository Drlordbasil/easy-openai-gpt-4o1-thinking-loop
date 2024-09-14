import json
import requests
import hashlib
import os
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor, as_completed

class WebResearchAndScraper:
    def __init__(self, structured_response_generator, cache_dir='./cache'):
        self.structured_response_generator = structured_response_generator
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_search_terms(self, topic):
        search_terms_schema = {
            "type": "object",
            "properties": {
                "search_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of search terms related to the topic"
                }
            },
            "required": ["search_terms"]
        }
         
        messages = [
            {"role": "system", "content": "You are an AI assistant that generates relevant search terms for a given topic."},
            {"role": "user", "content": f"Generate a list of 3 search terms for the topic: {topic}"}
        ]
        print("Generating search terms...")
        response = self.structured_response_generator.generate(messages, search_terms_schema)
        print("Search terms generated.")
        return response['search_terms']

    def search_and_scrape(self, search_terms, num_results=3):
        all_content = []
        print("Searching and scraping...")
        
        with ThreadPoolExecutor(max_workers=len(search_terms) * 2) as executor:
            future_to_url = {}
            for term in search_terms:
                future_to_url.update(self._search_google(term, num_results, executor))
                future_to_url.update(self._search_bing(term, num_results, executor))
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    if content:
                        all_content.append(content)
                except Exception as e:
                    print(f"Error scraping {url}: {e}")
        
        print("All content scraped.")
        return all_content

    def _search_google(self, query, num_results, executor):
        url = f"https://www.google.com/search?q={query}&num={num_results}"
        return self._search_engine(url, 'div', 'yuRUbf', executor)

    def _search_bing(self, query, num_results, executor):
        url = f"https://www.bing.com/search?q={query}&count={num_results}"
        return self._search_engine(url, 'li', 'b_algo', executor)

    def _search_engine(self, url, tag, class_name, executor):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [a.find('a')['href'] for a in soup.find_all(tag, class_=class_name) if a.find('a')]
            return {executor.submit(self._scrape_website, link): link for link in links}
        except Exception as e:
            print(f"Error searching {url}: {e}")
            return {}

    def _scrape_website(self, url):
        cache_key = self._get_cache_key(url)
        cached_content = self._get_cached_content(cache_key)
        if cached_content:
            return cached_content

        try:
            with webdriver.Chrome(options=self.chrome_options) as driver:
                driver.get(url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                text = soup.get_text()
                content = text[:1000]  # Limit to first 1000 characters
                self._cache_content(cache_key, content)
                return content
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def _get_cache_key(self, url):
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cached_content(self, cache_key):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def _cache_content(self, cache_key, content):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def summarize_research(self, content):
        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A concise summary of the research findings"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key points extracted from the research"
                }
            },
            "required": ["summary", "key_points"]
        }

        messages = [
            {"role": "system", "content": "You are an AI assistant that summarizes web research findings."},
            {"role": "user", "content": f"Summarize the following web research content:\n\n{' '.join(content)}"}
        ]

        return self.structured_response_generator.generate(messages, summary_schema)

    def conduct_research(self, topic):
        search_terms = self.generate_search_terms(topic)
        content = self.search_and_scrape(search_terms)
        if not content:
            return {"summary": "No content found", "key_points": []}
        return self.summarize_research(content)