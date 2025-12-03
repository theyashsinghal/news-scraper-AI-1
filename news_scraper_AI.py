# ==============================================================================
# --- GLOBAL USER SETTINGS ---
#
# How many articles to get from each source (e.g., 20)
MAX_ARTICLES_PER_SOURCE = 20
#
# --- CONCURRENCY SETTING (FIXED FOR STABILITY) ---
# How many articles from a single source should be fetched simultaneously.
# Reduced to 2 to minimize memory pressure and risk of Segmentation Faults.
MAX_CONCURRENT_ARTICLES_PER_SOURCE = 2
#
# --- PROXY CONFIGURATION ---
PROXY_SETTINGS = {
    "use_proxies": False,
    "proxy_url": None
}
# ==============================================================================

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import trafilatura  
import time 
import logging
import sqlite3
from datetime import datetime
import random 
import os 
import uuid 

# --- Imports for Parallelism and AI ---
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import threading

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

# --- Imports for Selenium ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import WebDriverException, TimeoutException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
# ---------------------------

# --- Configure logging ---
logging.basicConfig(filename='news_scraper.log',
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- GLOBAL LOCKS ---
db_lock = threading.Lock()
# NEW: Dedicated lock for the AI model to prevent concurrent memory access/Seg Faults
ai_lock = threading.Lock() 

# ==============================================================================
# --- SESSION, HEADER, and SELENIUM SETUP (Unchanged) ---
# ==============================================================================

def create_robust_session():
    logging.info("Creating new robust session with 3 retries on 5xx/connection/read errors.")
    session = requests.Session()
    retry_strategy = Retry(
        total=3, backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"], connect=True, read=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

BASE_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br', 'DNT': '1', 'Upgrade-Insecure-Requests': '1',
}

BROWSER_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
]

def get_headers(header_type):
    headers = BASE_HEADERS.copy()
    core_type = header_type.replace('requests_', '')
    if core_type == 'browser':
        headers['User-Agent'] = random.choice(BROWSER_USER_AGENTS)
    elif core_type == 'googlebot':
        headers['User-Agent'] = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
    elif core_type == 'feedfetcher':
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; FeedFetcher-Google; +http://www.google.com/feedfetcher.html)'}
    return headers

def create_selenium_driver():
    if not SELENIUM_AVAILABLE:
        logging.error("Cannot create Selenium driver, library not found.")
        return None
        
    logging.info("Initializing headless Selenium Chrome driver...")
    
    try:
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={random.choice(BROWSER_USER_AGENTS)}")
        
        if PROXY_SETTINGS["use_proxies"] and PROXY_SETTINGS["proxy_url"]:
            logging.info(f"Configuring Selenium driver to use proxy.")
            options.add_argument(f"--proxy-server={PROXY_SETTINGS['proxy_url']}")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(60)
        logging.info("Selenium driver initialized successfully.")
        return driver
    except Exception as e:
        logging.critical(f"Failed to initialize Selenium driver: {e}")
        return None

# ==============================================================================
# --- SOURCE CONFIGURATION (Unchanged) ---
# ==============================================================================

SOURCE_CONFIG = [
    {
        'name': 'BBC',
        'rss_url': 'http://feeds.bbci.co.uk/news/world/rss.xml',
        'rss_headers_type': 'feedfetcher',
        'article_strategies': ['requests_browser'],
        'article_url_contains': None,
        'referer': 'https://www.bbc.com/news',
    },
    {
        'name': 'Times of India',
        'rss_url': 'https://timesofindia.indiatimes.com/rssfeeds/296589292.cms',
        'rss_headers_type': 'feedfetcher',
        'article_strategies': ['selenium_browser'],
        'article_url_contains': '.cms',
        'referer': 'https://timesofindia.indiatimes.com/',
    },
    {
        'name': 'The Guardian',
        'rss_url': 'https://www.theguardian.com/world/rss',
        'rss_headers_type': 'feedfetcher',
        'article_strategies': ['requests_browser'],
        'article_url_contains': None,
        'referer': 'https://www.theguardian.com/',
    },
    {
        'name': 'The Hindu',
        'rss_url': 'https://www.thehindu.com/news/national/feeder/default.rss',
        'rss_headers_type': 'browser',
        'article_strategies': ['selenium_browser'],
        'article_url_contains': None,
        'referer': 'https://www.thehindu.com/',
    }
]

# ==============================================================================
# --- AI/DB SETUP (Model Loading Unchanged) ---
# ==============================================================================

semantic_model = None
if SentenceTransformer is not None:
    try:
        logging.info("Loading AI Semantic Model (all-MiniLM-L6-v2)...")
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("AI Model loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load AI model: {e}. Clustering is disabled.")
        semantic_model = None

db_path = 'news_articles.db'
logging.info(f"Initializing database connection at: {db_path}")
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    cluster_id TEXT, 
    source TEXT,
    title TEXT, 
    url TEXT UNIQUE, 
    summary TEXT, 
    image_url TEXT, 
    scraped_at TIMESTAMP
)
''')

# --- MIGRATION: Check if 'cluster_id' exists, if not, add it ---
cursor.execute("PRAGMA table_info(news)")
columns = [info[1] for info in cursor.fetchall()]
if 'cluster_id' not in columns:
    logging.info("Performing database migration: Adding 'cluster_id' column.")
    try:
        cursor.execute("ALTER TABLE news ADD COLUMN cluster_id TEXT")
    except Exception as e:
        logging.error(f"Migration failed: {e}")
conn.commit()
# ---------------------------------------------------------------

# ==============================================================================
# --- AI DEDUPLICATION AND SAVE LOGIC (AI LOCK INTEGRATED) ---
# ==============================================================================

def get_cluster_id_for_article(new_title, new_summary):
    """
    Checks DB for semantically similar articles and returns a cluster_id.
    Uses ai_lock to protect the SentenceTransformer model.
    """
    if semantic_model is None or util is None:
        return str(uuid.uuid4())

    try:
        # Acquire AI lock to prevent parallel embedding calculation
        with ai_lock:
            # NOTE: DB select is fine outside the DB lock here as no write operation follows immediately
            cursor.execute('''
                SELECT title, summary, cluster_id 
                FROM news 
                WHERE scraped_at >= datetime('now', '-1 day')
            ''')
            recent_articles = cursor.fetchall()

            if not recent_articles: return str(uuid.uuid4()) 

            existing_texts = [f"{row[0]}. {row[1]}" for row in recent_articles]
            existing_ids = [row[2] for row in recent_articles]
            new_text = f"{new_title}. {new_summary}"

            # CRITICAL SECTION: AI Model Encoding
            existing_embeddings = semantic_model.encode(existing_texts, convert_to_tensor=True)
            new_embedding = semantic_model.encode(new_text, convert_to_tensor=True)
        
        # Release AI lock; calculation is safe
        cosine_scores = util.cos_sim(new_embedding, existing_embeddings)[0]

        best_score = -1
        best_idx = -1
        for i, score in enumerate(cosine_scores):
            if score > best_score:
                best_score = score
                best_idx = i

        THRESHOLD = 0.70
        if best_score >= THRESHOLD:
            logging.info(f"DEDUPLICATION: Found match (Score: {best_score:.2f}). Linking to Cluster ID: {existing_ids[best_idx]}")
            return existing_ids[best_idx]
        else:
            return str(uuid.uuid4())
            
    except Exception as e:
        logging.error(f"Error during AI clustering calculation: {e}")
        return str(uuid.uuid4())


def save_article(source, title, url, summary, image_url):
    """
    Saves a single article to the SQLite database with a guaranteed final URL check.
    """
    try:
        # --- ROBUST CLEANING ---
        title = " ".join(title.replace('\n', ' ').replace('\r', ' ').split()).strip()
        if summary:
            summary_lines = [line.strip() for line in summary.splitlines() if line.strip()]
            summary = "\n\n".join(summary_lines)
        if not summary: summary = "No content available"
        if not image_url: image_url = "No image available"
        
        # 1. AI STEP (outside db_lock): Calculate cluster ID.
        cluster_id = get_cluster_id_for_article(title, summary)

        # 2. CRITICAL FINAL CHECK AND INSERT
        with db_lock:
            # RE-CHECK: Catch race condition (cross-source or between filter/save)
            cursor.execute("SELECT id FROM news WHERE url = ?", (url,))
            if cursor.fetchone():
                logging.warning(f"RACE CONDITION AVOIDED: Skipped insertion of {title} after AI check.")
                return False
            
            # 3. INSERT is now safe
            cursor.execute('''
                INSERT INTO news (cluster_id, source, title, url, summary, image_url, scraped_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (cluster_id, source, title, url, summary, image_url, datetime.now()))
            conn.commit()
        
        logging.info(f"Saved article: {title} from {source}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving article {title}: {e}")
        return False

# ==============================================================================
# --- ARTICLE-LEVEL PARALLEL PROCESSOR ---
# ==============================================================================

def process_single_article(item, source_config, session, selenium_driver, proxies_dict):
    """
    Handles the request, extraction, and save for one article. 
    """
    name = source_config['name']
    
    if not item.link: return (False, name)

    article_url = item.link.text.strip()
    rss_title = item.title.text if item.title else "Title not found"
    
    summary = None
    raw_html = None
    final_title = rss_title
    image_url = "No image available"
    strategies = source_config['article_strategies']
    
    for i, strategy in enumerate(strategies):
        try:
            logging.info(f"[{name}] Article: {article_url}. Trying '{strategy}'...")
            
            # --- STRATEGY ROUTER ---
            if strategy.startswith('requests_'):
                header_type = strategy.replace('requests_', '')
                article_headers = get_headers(header_type)
                article_headers['Referer'] = source_config['referer']
                page_response = session.get(article_url, headers=article_headers, timeout=10, proxies=proxies_dict)
                page_response.raise_for_status()
                raw_html = page_response.text
            
            elif strategy == 'selenium_browser':
                if not selenium_driver: raise Exception("Selenium driver not available.")
                selenium_driver.get(article_url)
                WebDriverWait(selenium_driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "p")))
                raw_html = selenium_driver.page_source

            # Extraction
            temp_summary = trafilatura.extract(raw_html, include_comments=False, include_tables=False)
            word_count = len(temp_summary.split()) if temp_summary else 0
            
            if word_count >= 80:
                summary = temp_summary
                soup = BeautifulSoup(raw_html, 'html.parser')
                page_title = soup.find('title')
                if page_title: final_title = page_title.text
                og_image = soup.find('meta', property='og:image')
                if og_image: image_url = og_image['content']
                logging.info(f"[{name}] Success with '{strategy}'. ({word_count} words).")
                break

            else:
                logging.warning(f"[{name}] FAILED with '{strategy}' (content too short: {word_count} words).")

        except Exception as e:
            logging.error(f"[{name}] Strategy '{strategy}' failed on {article_url}: {e}")
            if strategy == 'selenium_browser' and selenium_driver: 
                 selenium_driver.refresh()
            time.sleep(random.uniform(0.5, 1.0))

    # 5. Fallback Logic
    if not summary:
        logging.error(f"[{name}] All strategies failed. Falling back to RSS description.")
        if item.description:
            summary_soup = BeautifulSoup(item.description.text, 'html.parser')
            summary = summary_soup.get_text().strip()
        else:
            summary = "No content available"

    # 6. Save (Uses the revised save_article with final RACE CONDITION check)
    was_saved = save_article(name, final_title, article_url, summary, image_url)
    time.sleep(random.uniform(0.5, 1.5))
    
    return (was_saved, name)


# ==============================================================================
# --- SOURCE-LEVEL PROCESSOR (Pre-filter to prevent concurrent processing) ---
# ==============================================================================

def scrape_source(session, selenium_driver, source_config, proxies_dict):
    """
    Scrapes one source. Performs the duplicate check at the RSS stage to prevent
    submitting duplicates to the parallel executor.
    """
    name = source_config['name']
    rss_url = source_config['rss_url']
    saved_count = 0
    
    logging.info(f"Starting scrape for {name} RSS feed. Article Concurrency: {MAX_CONCURRENT_ARTICLES_PER_SOURCE}")
    
    try:
        # 1. Get RSS Feed
        rss_headers = get_headers(source_config['rss_headers_type'])
        response = session.get(rss_url, headers=rss_headers, timeout=15, proxies=proxies_dict)  
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.content, 'xml')
        all_items = soup.find_all('item')[:MAX_ARTICLES_PER_SOURCE]
        
        items_to_process = []
        
        # 2. CRITICAL PRE-CHECK: Filter out known duplicates here
        logging.info(f"[{name}] Found {len(all_items)} items. Performing duplicate pre-check...")
        for item in all_items:
            if not item.link: continue
            article_url = item.link.text.strip()
            
            # --- THREAD-SAFE CHECK ---
            with db_lock:
                cursor.execute("SELECT id FROM news WHERE url = ?", (article_url,))
                if cursor.fetchone():
                    continue # Skip it, don't submit to the parallel queue
            
            # Check for URL filter
            if source_config['article_url_contains'] and source_config['article_url_contains'] not in article_url:
                logging.warning(f"[{name}] Skipping non-article link during pre-check: {article_url}")
                continue

            items_to_process.append(item)
            
        logging.info(f"[{name}] {len(items_to_process)} unique articles remaining. Submitting to article-level executor.")

        # 3. Use a local ThreadPoolExecutor for concurrent articles
        local_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ARTICLES_PER_SOURCE)
        
        futures = []
        for item in items_to_process:
            future = local_executor.submit(
                process_single_article, 
                item, 
                source_config, 
                session, 
                selenium_driver, 
                proxies_dict
            )
            futures.append(future)

        # 4. Collect results as they complete
        for future in as_completed(futures, timeout=300): 
            try:
                was_saved, _ = future.result()
                if was_saved:
                    saved_count += 1
            except Exception as e:
                logging.error(f"[{name}] Article thread failed: {e}")
        
        local_executor.shutdown() 

    except requests.RequestException as e:
        logging.error(f"Failed to fetch {name} RSS feed: {e}")
    except Exception as e:
        logging.error(f"Failed to process {name} source: {e}")
        
    return (name, saved_count)

# ==============================================================================
# --- MAIN EXECUTION AND CLEANUP ---
# ==============================================================================

def scrape_source_wrapper(source, session, proxies_dict):
    name = source.get('name', 'Unknown')
    driver = None
    needs_selenium = any('selenium' in s for s in source.get('article_strategies', []))
    
    try:
        if needs_selenium and SELENIUM_AVAILABLE:
            driver = create_selenium_driver()
            if not driver:
                logging.error(f"[{name}] Selenium driver failed to start.")
        
        return scrape_source(session, driver, source, proxies_dict)
    
    except Exception as e:
        logging.critical(f"--- CRITICAL: Scrape job for {name} failed entirely. --- {e}")
        return (name, 0)
    
    finally:
        if driver:
            logging.info(f"[{name}] (Thread) Finished. Attempting to shut down its Selenium driver.")
            pid_to_kill = None
            try: pid_to_kill = driver.service.process.pid
            except Exception: pass
            
            try: driver.quit()
            except Exception as e:
                logging.warning(f"[{name}] driver.quit() failed: {e}. Attempting surgical kill.")
                if pid_to_kill:
                    try: os.kill(pid_to_kill, 9)
                    except Exception as e_kill: logging.error(f"[{name}] Failed to kill process: {e_kill}")
            

def scrape_all():
    logging.info("--- Starting new scraping job (Source Parallel Mode) ---")
    session = create_robust_session()
    proxies_dict = None
    if PROXY_SETTINGS["use_proxies"] and PROXY_SETTINGS["proxy_url"]:
        proxies_dict = {"http": PROXY_SETTINGS["proxy_url"], "https": PROXY_SETTINGS["proxy_url"]}
    
    all_counts = {}
    total_saved = 0
    futures = []
    executor = ThreadPoolExecutor(max_workers=len(SOURCE_CONFIG))

    try:
        for source in SOURCE_CONFIG:
            future = executor.submit(scrape_source_wrapper, source, session, proxies_dict)
            futures.append(future)

        logging.info(f"Submitted {len(futures)} source jobs. Waiting up to 300s...")
        done, not_done = wait(futures, timeout=300)

        for future in done:
            try:
                name, count = future.result()
                all_counts[name] = count
                total_saved += count
            except Exception as e:
                logging.error(f"A future source job resulted in an error: {e}")
        
        if not_done:
            logging.critical(f"--- TIMEOUT: {len(not_done)} source jobs did not complete in 300s. ---")
            for future in not_done:
                logging.error("A thread has timed out and will be abandoned.")
                all_counts["Timed_Out_Jobs"] = all_counts.get("Timed_Out_Jobs", 0) + 1

    except Exception as e:
        logging.critical(f"--- CRITICAL: The entire scrape_all job failed. --- {e}")
        
    finally:
        logging.info("Shutting down thread pool (wait=False)...")
        executor.shutdown(wait=False)
        
        log_summary = ", ".join(f"{count} {name}" for name, count in all_counts.items())
        log_message = f"Scraped: {log_summary} articles. (Total saved: {total_saved})"
        
        logging.info(log_message)
        print(log_message)
        logging.info("--- Scraping job finished ---")


def main():
    global conn
    try:
        logging.info("--- Scraper service started (CI Mode: Run Once) ---")
        print("Running single scrape for CI...")
        scrape_all()
        print("Scrape finished.")
    except Exception as e:
        logging.critical(f"A critical error occurred in the main function: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("--- Scraper service stopped and database connection closed. ---")
            print("Scraper stopped and database connection closed.")
        
        logging.info("--- Main thread finished. Forcing process exit to kill zombie threads. ---")
        os._exit(0)

if __name__ == '__main__':
    main()
