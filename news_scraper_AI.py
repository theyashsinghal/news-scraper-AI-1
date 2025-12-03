# ==============================================================================
# --- GLOBAL USER SETTINGS ---
#
# How many articles to get from each source (e.g., 20)
MAX_ARTICLES_PER_SOURCE = 10
#
# NOTE: Article concurrency setting removed. Articles are processed sequentially.
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
import re # --- NEW: Import regex for title cleanup

# --- Imports for Parallelism and AI ---
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import threading

# NOTE: AI dependencies remain imported, but usage is restricted to single-thread functions
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
# Removed ai_lock as clustering is now decoupled and run sequentially.

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
# --- AI/DB SETUP and MIGRATION (Model Loading Unchanged) ---
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
# --- FINAL CLUSTERING LOGIC (FIXED FOR TITLE CLEANUP) ---
# ==============================================================================

# Regex pattern to clean up titles by removing common separators and metadata/source names
TITLE_CLEANUP_PATTERN = re.compile(r'\s*([|—–\-:.]\s*.*)$')

def clean_title_for_clustering(title):
    """Removes common source/category metadata from the end of a title."""
    match = TITLE_CLEANUP_PATTERN.search(title)
    if match:
        # If the separator is found, return only the part before it
        return title[:match.start()].strip()
    return title.strip()


def finalize_clustering():
    """
    Runs clustering logic on all articles in the last 24 hours. 
    It now cleans titles before embedding for better accuracy.
    """
    if semantic_model is None or util is None:
        logging.warning("AI Model not available. Skipping final clustering step.")
        return

    logging.info("Starting single-threaded final clustering process...")
    
    with db_lock:
        # Fetch ALL articles (clustered and unclustered) from the last 24h
        cursor.execute('''
            SELECT id, title, summary, cluster_id 
            FROM news 
            WHERE scraped_at >= datetime('now', '-1 day')
        ''')
        all_articles = cursor.fetchall()

    if not all_articles:
        logging.info("No recent articles found for clustering.")
        return

    # 1. Prepare data structures: 
    article_db_id_to_cluster_id = {row[0]: row[3] for row in all_articles if row[3]}
    article_db_id_to_index = {row[0]: i for i, row in enumerate(all_articles)}
    
    # NEW: Clean titles before combining with summary for embedding
    all_texts = [
        f"{clean_title_for_clustering(row[1])}. {row[2]}" 
        for row in all_articles
    ]
    all_db_ids = [row[0] for row in all_articles]

    logging.info(f"Clustering {len(all_articles)} total articles...")
    
    try:
        # 2. Generate Embeddings (Single-threaded, safe)
        all_embeddings = semantic_model.encode(all_texts, convert_to_tensor=True)
        
        updates_needed = {}
        THRESHOLD = 0.70

        # 3. Iterate through all articles and find the best match for unclustered ones
        for i, article in enumerate(all_articles):
            db_id = article[0]
            current_cluster_id = article[3]
            
            # Only process articles that haven't been clustered yet
            if current_cluster_id is not None:
                continue

            candidate_emb = all_embeddings[i]
            
            # --- Perform Comparison against ALL other articles ---
            cosine_scores = util.cos_sim(candidate_emb, all_embeddings)[0]
            
            best_score = -1.0
            best_match_id = None

            for j, score_tensor in enumerate(cosine_scores):
                score = score_tensor.item()
                match_id = all_db_ids[j]

                # Skip self-comparison and already processed matches that have been updated in this loop
                if db_id == match_id:
                    continue

                if score > best_score and score >= THRESHOLD:
                    best_score = score
                    best_match_id = match_id

            # 4. Assign Cluster ID
            if best_match_id is not None:
                matched_article_cluster_id = article_db_id_to_cluster_id.get(best_match_id)
                
                if matched_article_cluster_id:
                    # Match found with an existing cluster or an article already assigned a cluster in this pass
                    updates_needed[db_id] = matched_article_cluster_id
                else:
                    # Match found with another *new* article (create new cluster for both)
                    assigned_cluster_id = updates_needed.get(best_match_id)
                    
                    if assigned_cluster_id:
                        updates_needed[db_id] = assigned_cluster_id
                    else:
                        new_cluster_id = str(uuid.uuid4())
                        updates_needed[db_id] = new_cluster_id
                        
                        # Assign to the matching article as well, if it's new
                        if all_articles[article_db_id_to_index[best_match_id]][3] is None:
                            updates_needed[best_match_id] = new_cluster_id
                        
                # Update the local cluster map to ensure future candidates check against this new cluster
                article_db_id_to_cluster_id[db_id] = updates_needed[db_id]
                
            else:
                # No match found, assign a new cluster ID
                updates_needed[db_id] = str(uuid.uuid4())
                article_db_id_to_cluster_id[db_id] = updates_needed[db_id]


        # 5. Apply Updates to DB
        if updates_needed:
            with db_lock:
                for db_id, cluster_id in updates_needed.items():
                    cursor.execute(
                        "UPDATE news SET cluster_id = ? WHERE id = ?", 
                        (cluster_id, db_id)
                    )
                conn.commit()
        
        logging.info(f"Clustering complete. Assigned {len(updates_needed)} cluster IDs.")

    except Exception as e:
        logging.error(f"Critical error during final clustering: {e}")

# ==============================================================================
# --- SAVE LOGIC (Unchanged - saves NULL cluster_id) ---
# ==============================================================================

def save_article(source, title, url, summary, image_url):
    """
    Saves a single article *without* calculating the cluster_id during the scrape.
    """
    try:
        # --- ROBUST CLEANING ---
        title = " ".join(title.replace('\n', ' ').replace('\r', ' ').split()).strip()
        if summary:
            summary_lines = [line.strip() for line in summary.splitlines() if line.strip()]
            summary = "\n\n".join(summary_lines)
        if not summary: summary = "No content available"
        if not image_url: image_url = "No image available"
        
        # --- CRITICAL CHECK AND INSERT ---
        with db_lock:
            # RE-CHECK: Catch race condition
            cursor.execute("SELECT id FROM news WHERE url = ?", (url,))
            if cursor.fetchone():
                logging.warning(f"RACE CONDITION AVOIDED: Skipped insertion of {title}.")
                return False
            
            # Insert with NULL cluster_id for later processing
            cursor.execute('''
                INSERT INTO news (cluster_id, source, title, url, summary, image_url, scraped_at) 
                VALUES (NULL, ?, ?, ?, ?, ?, ?)
            ''', (source, title, url, summary, image_url, datetime.now()))
            conn.commit()
        
        logging.info(f"Saved article: {title} from {source}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving article {title}: {e}")
        return False

# ==============================================================================
# --- ARTICLE PROCESSING (Sequential Logic Integrated) ---
# ==============================================================================

def scrape_source(session, selenium_driver, source_config, proxies_dict):
    """
    Scrapes one source. Articles are processed sequentially within this dedicated thread.
    """
    name = source_config['name']
    rss_url = source_config['rss_url']
    saved_count = 0
    
    logging.info(f"Starting scrape for {name} RSS feed (Sequential Article Processing)...")
    
    try:
        # 1. Get RSS Feed
        rss_headers = get_headers(source_config['rss_headers_type'])
        response = session.get(rss_url, headers=rss_headers, timeout=15, proxies=proxies_dict)  
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.content, 'xml')
        all_items = soup.find_all('item')[:MAX_ARTICLES_PER_SOURCE]
        
        logging.info(f"[{name}] Found {len(all_items)} items. Starting sequential processing...")
        
        # 2. Sequential Article Processing Loop
        for item in all_items:
            
            if not item.link: continue
            article_url = item.link.text.strip()
            rss_title = item.title.text if item.title else "Title not found"
            
            # --- CRITICAL PRE-CHECK (1): Filter out known duplicates here
            with db_lock:
                cursor.execute("SELECT id FROM news WHERE url = ?", (article_url,))
                if cursor.fetchone():
                    continue # Skip it
            
            # Check for URL filter
            if source_config['article_url_contains'] and source_config['article_url_contains'] not in article_url:
                logging.warning(f"[{name}] Skipping non-article link during pre-check: {article_url}")
                continue
            
            # --- ARTICLE SCRAPING LOGIC ---
            summary = None
            raw_html = None
            final_title = rss_title
            image_url = "No image available"
            strategies = source_config['article_strategies']
            
            for i, strategy in enumerate(strategies):
                try:
                    logging.info(f"[{name}] Article: {article_url}. Trying '{strategy}'...")
                    
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
                        break # Success
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

            # 6. Save (Sequential commit per article)
            was_saved = save_article(name, final_title, article_url, summary, image_url)
            if was_saved:
                saved_count += 1
            
            time.sleep(random.uniform(0.5, 1.5)) # Politeness delay
        
    except requests.RequestException as e:
        logging.error(f"Failed to fetch {name} RSS feed: {e}")
    except Exception as e:
        logging.error(f"Failed to process {name} source: {e}")
        
    return (name, saved_count)

# ==============================================================================
# --- MAIN EXECUTION AND CLEANUP (Source-Level Parallelism) ---
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
        
        # --- NEW STEP: Run AI Clustering SEQUENTIALLY after all scraping is done ---
        finalize_clustering()
        
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
