# ==============================================================================
# --- GLOBAL USER SETTINGS ---
#
# How many articles to get from each source (e.g., 25)
# This is a 'max' value. If a feed only has 20 articles, it will get 20.
MAX_ARTICLES_PER_SOURCE = 5
#
# --- NEW: PROXY CONFIGURATION ---
# Set 'use_proxies' to True to route all requests (Requests & Selenium)
# through the 'proxy_url'.
#
# 'proxy_url' should be in the format: http://username:password@proxy.example.com:8080
PROXY_SETTINGS = {
    "use_proxies": False,
    "proxy_url": None  # e.g., "http://user:pass@proxy.service.com:8080"
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
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import sys 

# --- NEW: Imports for AI/Semantics ---
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    logging.critical("sentence-transformers not installed. Run 'pip install sentence-transformers'. AI clustering will be skipped.")
    SentenceTransformer = None
    util = None
# -------------------------------------

# --- Imports for Selenium ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.common.exceptions import WebDriverException, TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    logging.critical("Selenium not installed. Run 'pip install selenium'. Selenium-dependent sources will fail.")
    SELENIUM_AVAILABLE = False
# ---------------------------

# --- Configure logging ---
logging.basicConfig(filename='news_scraper.log',
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Robust Session and Header Management ---

def create_robust_session():
    """Creates a requests.Session with automatic retries."""
    logging.info("Creating new robust session with 3 retries on 5xx/connection/read errors.")
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        connect=True,
        read=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Headers and User-Agents
BASE_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Upgrade-Insecure-Requests': '1',
}

BROWSER_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
]
GOOGLEBOT_USER_AGENT = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
FEEDFETCHER_USER_AGENT = 'Mozilla/5.0 (compatible; FeedFetcher-Google; +http://www.google.com/feedfetcher.html)'

def get_headers(header_type):
    """Returns a complete header dictionary for a given "persona"."""
    headers = BASE_HEADERS.copy()
    core_type = header_type.replace('requests_', '')

    if core_type == 'browser':
        headers['User-Agent'] = random.choice(BROWSER_USER_AGENTS)
    elif core_type == 'googlebot':
        headers['User-Agent'] = GOOGLEBOT_USER_AGENT
    elif core_type == 'feedfetcher':
        headers = {'User-Agent': FEEDFETCHER_USER_AGENT}
    return headers

def create_selenium_driver():
    """Initializes and returns a headless Selenium Chrome WebDriver."""
    if not SELENIUM_AVAILABLE:
        return None

    try:
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={random.choice(BROWSER_USER_AGENTS)}")

        if PROXY_SETTINGS["use_proxies"] and PROXY_SETTINGS["proxy_url"]:
            options.add_argument(f"--proxy-server={PROXY_SETTINGS['proxy_url']}")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(20)
        logging.info("Selenium driver initialized successfully.")
        return driver
    except WebDriverException as e:
        logging.critical(f"Failed to initialize Selenium driver. Error: {e}")
        return None
    except Exception as e:
        logging.critical(f"An unexpected error occurred during Selenium initialization: {e}")
        return None

# --- Central Source Configuration ---
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
# AI MODEL INITIALIZATION
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


# ==============================================================================
# DATABASE SETUP
# ==============================================================================
db_path = 'news_articles.db'
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
db_lock = threading.Lock() # Thread lock for database operations

# Create table and check/add 'cluster_id' column
try:
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
    cursor.execute("PRAGMA table_info(news)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'cluster_id' not in columns:
        cursor.execute("ALTER TABLE news ADD COLUMN cluster_id TEXT")
        logging.info("Migration successful: Added 'cluster_id' column.")
    conn.commit()
except Exception as e:
    logging.critical(f"Database setup failed: {e}")
    sys.exit(1)


# ==============================================================================
# AI DEDUPLICATION LOGIC
# ==============================================================================

def get_cluster_id_for_article(new_title, new_summary):
    """Checks DB for similar articles and assigns a cluster_id."""
    if semantic_model is None or util is None:
        return str(uuid.uuid4())

    try:
        # --- MODIFIED: Reverted to 21-day memory check ---
        # Fetch recent articles (last 21 day)
        cursor.execute('''
            SELECT title, summary, cluster_id 
            FROM news 
            WHERE scraped_at >= datetime('now', '-21 day')
        ''')
        # ------------------------------------------------
        recent_articles = cursor.fetchall()

        if not recent_articles:
            return str(uuid.uuid4())

        # Prepare data and convert to Embeddings
        existing_texts = [f"{row[0]}. {row[1]}" for row in recent_articles]
        existing_ids = [row[2] for row in recent_articles]
        new_text = f"{new_title}. {new_summary}"

        existing_embeddings = semantic_model.encode(existing_texts, convert_to_tensor=True)
        new_embedding = semantic_model.encode(new_text, convert_to_tensor=True)

        # Calculate Cosine Similarity
        cosine_scores = util.cos_sim(new_embedding, existing_embeddings)[0]

        best_score = -1
        best_idx = -1
        for i, score in enumerate(cosine_scores):
            if score > best_score:
                best_score = score
                best_idx = i

        THRESHOLD = 0.70 # Default semantic threshold
        
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
    Saves a single article to the database. 
    INCLUDES: Smart Paragraph Preservation & 90-word minimum check.
    """
    
    # --- STEP 0: STRICT GLOBAL WORD COUNT CHECK ---
    # We use simple whitespace splitting to count words, 
    # which works for both flat text and text with newlines.
    if not summary:
        final_word_count = 0
    else:
        final_word_count = len(summary.split())

    MIN_SUMMARY_WORDS = 90
    if final_word_count < MIN_SUMMARY_WORDS:
        logging.warning(f"SKIPPED (GLOBAL WORD LIMIT): Article '{title}' from {source} has only {final_word_count} words (Min: {MIN_SUMMARY_WORDS}).")
        return False
    # ---------------------------------------------
    
    try:
        # --- SMART CLEANING: PRESERVE STRUCTURE ---
        # 1. Clean Title: Titles should generally be one line, so we flatten them.
        title = " ".join(title.split()).strip()

        # 2. Clean Summary: Preserve Paragraphs (\n)
        if summary:
            # Split the text into lines based on existing newlines
            lines = summary.splitlines()
            
            # Clean each line individually to remove extra spaces *inside* the sentence
            # but keep the line itself valid.
            cleaned_lines = [" ".join(line.split()) for line in lines]
            
            # Join the lines back together with DOUBLE newlines to create clear paragraphs.
            # We filter out empty lines to avoid huge gaps.
            summary = "\n\n".join([line for line in cleaned_lines if line])
        # ------------------------------------------
        
        if not image_url:
            image_url = "No image available"

        # --- THREAD-SAFE BLOCK ---
        with db_lock:
            # First, check if the URL already exists (the primary skip)
            cursor.execute("SELECT id FROM news WHERE url = ?", (url,))
            if cursor.fetchone():
                logging.info(f"Duplicate article skipped: {title} from {source}")
                return False 
            
            # --- Get Cluster ID ---
            # Use the already cleaned/validated summary for clustering
            cluster_id = get_cluster_id_for_article(title, summary)
            # ----------------------

            # If not found, insert it with the cluster_id
            cursor.execute('''
                INSERT INTO news (cluster_id, source, title, url, summary, image_url, scraped_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (cluster_id, source, title, url, summary, image_url, datetime.now()))
            conn.commit()
        # -------------------------
        
        logging.info(f"Saved article: {title} from {source} ({final_word_count} words)")
        return True
        
    except Exception as e:
        logging.error(f"Error saving article {title}: {e}")
        return False


# --- RE-ARCHITECTED: Generic Scraper Function with Strategy Loop ---
def scrape_source(session, selenium_driver, source_config, proxies_dict):
    """
    A generic function that scrapes any source based on its config.
    It will try every strategy in `article_strategies` to get the full text
    before falling back to the RSS summary.
    """
    name = source_config['name']
    rss_url = source_config['rss_url']
    
    articles_saved_list = []
    
    logging.info(f"Starting scrape for {name} RSS feed: {rss_url}")
    
    try:
        # 1. Get RSS Feed
        rss_headers = get_headers(source_config['rss_headers_type'])
        response = session.get(rss_url, headers=rss_headers, timeout=20, proxies=proxies_dict)  
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        logging.info(f"Found {len(items)} articles in {name} RSS feed. Processing up to {MAX_ARTICLES_PER_SOURCE}.")

# 2. Process each article
        # UPDATED: Loop through ALL items, not just the first 20
        for item in items:
            
        # UPDATED: Stop processing if we have already saved 20 new articles
        if len(articles_saved_list) >= MAX_ARTICLES_PER_SOURCE:
             logging.info(f"[{name}] Target reached: Successfully saved {MAX_ARTICLES_PER_SOURCE} new articles.")
              break

          article_url = None
          rss_title = "Title not found"
          rss_description = None

            try:
                if not item.link:
                    continue
                
                article_url = item.link.text.strip()
                
                # Check for URL filter
                if source_config['article_url_contains'] and source_config['article_url_contains'] not in article_url:
                    logging.warning(f"[{name}] Skipping non-article link: {article_url}")
                    continue
                
                # --- Early Skip Check ---
                with db_lock:
                    cursor.execute("SELECT id FROM news WHERE url = ?", (article_url,))
                    if cursor.fetchone():
                        logging.info(f"[{name}] Early skip: URL {article_url} already exists.")
                        time.sleep(random.uniform(0.1, 0.3))
                        continue
                # -------------------------

                rss_title = item.title.text if item.title else "Title not found"
                
                # Get the raw RSS description now, for potential fallback later
                if item.description:
                    # Use BeautifulSoup to strip any HTML from the RSS description
                    summary_soup = BeautifulSoup(item.description.text, 'html.parser')
                    rss_description = summary_soup.get_text().strip()
                
                # --- NEW MULTI-STRATEGY LOGIC ---
                summary = None
                raw_html = None
                final_title = rss_title
                image_url = "No image available"
                
                strategies = source_config['article_strategies']
                
                for i, strategy in enumerate(strategies):
                    logging.info(f"[{name}] Article: {article_url}")
                    logging.info(f"[{name}] Attempt {i+1}/{len(strategies)}: Trying with '{strategy}' strategy...")
                    
                    try:
                        # --- STRATEGY ROUTER ---
                        if strategy.startswith('requests_'):
                            header_type = strategy.replace('requests_', '')
                            article_headers = get_headers(header_type)
                            article_headers['Referer'] = source_config['referer']
                            
                            page_response = session.get(article_url, headers=article_headers, timeout=20, proxies=proxies_dict)
                            page_response.raise_for_status()
                            raw_html = page_response.text
                        
                        elif strategy == 'selenium_browser':
                            if not selenium_driver:
                                logging.error(f"[{name}] Selenium strategy selected but driver is not available. Skipping.")
                                continue
                            
                            selenium_driver.get(article_url)
                            
                            # Use Explicit Wait
                            try:
                                WebDriverWait(selenium_driver, 10).until(
                                    EC.presence_of_element_located((By.TAG_NAME, "p"))
                                )
                                logging.info(f"[{name}] Page content loaded.")
                            except TimeoutException:
                                logging.warning(f"[{name}] Page timed out (10s). Proceeding anyway.")
                                
                            raw_html = selenium_driver.page_source
                        
                        else:
                            logging.error(f"[{name}] Unknown strategy: {strategy}. Skipping.")
                            continue
                        # --- END STRATEGY ROUTER ---

                        # 4. Extract Content
                        if not raw_html:
                            logging.warning(f"[{name}] FAILED with '{strategy}' (HTML was empty).")
                            continue

                        temp_summary = trafilatura.extract(raw_html, include_comments=False, include_tables=False)
                        
                        word_count = len(temp_summary.split()) if temp_summary else 0
                        
                        # --- MODIFIED: Word count check set to 90 ---
                        if word_count >= 90:
                        # --------------------------------------------
                            logging.info(f"[{name}] Success with '{strategy}'. Found content ({word_count} words).")
                            summary = temp_summary
                            
                            # Parse metadata
                            soup = BeautifulSoup(raw_html, 'html.parser')
                            page_title = soup.find('title')
                            if page_title:
                                final_title = page_title.text
                                
                            og_image = soup.find('meta', property='og:image')
                            if og_image:
                                image_url = og_image['content']
                                
                            break # <-- Success! Exit the strategy loop.
                        else:
                            logging.warning(f"[{name}] FAILED with '{strategy}' (content was too short: {word_count} words).")
                    
                    except Exception as e:
                        logging.error(f"[{name}] Request failed for strategy '{strategy}' on URL {article_url}: {e}")
                        
                    if i < len(strategies) - 1:
                        time.sleep(random.uniform(0.5, 1.0))
                
                # --- END OF STRATEGY LOOP ---

                # 5. Final Summary Assignment (If all strategies failed, use the RSS description)
                if not summary:
                    logging.error(f"[{name}] All scrape strategies failed for {article_url}. Falling back to RSS description.")
                    summary = rss_description or "No content available"

                # 6. Save (The save_article function now handles the final 90-word check)
                was_saved = save_article(name, final_title, article_url, summary, image_url)
                if was_saved:
                    articles_saved_list.append(final_title)
                    
                time.sleep(random.uniform(0.5, 1.5))

            except Exception as e:
                logging.error(f"[{name}] Article-level Error: {e} for url {article_url}")

    except requests.RequestException as e:
        logging.error(f"Failed to fetch {name} RSS feed: {e}")
    except Exception as e:
        logging.error(f"Failed to parse {name} RSS feed: {e}")
        
    return (name, len(articles_saved_list))


# --- NEW: Thread Wrapper Function ---
def scrape_source_wrapper(source, session, proxies_dict):
    """
    A wrapper function to be run in a separate thread.
    It creates and destroys its own Selenium driver if needed.
    """
    name = source.get('name', 'Unknown')
    driver = None
    
    needs_selenium = any('selenium' in s for s in source.get('article_strategies', []))
    
    try:
        if needs_selenium and SELENIUM_AVAILABLE:
            logging.info(f"[{name}] (Thread) requires Selenium. Initializing driver...")
            driver = create_selenium_driver()
            if not driver:
                logging.error(f"[{name}] (Thread) Selenium driver failed to start. This source will fail.")
        
        return scrape_source(session, driver, source, proxies_dict)
    
    except Exception as e:
        logging.critical(f"--- CRITICAL: (Thread) Scrape job for {name} failed entirely. --- {e}")
        return (name, 0)
    
    finally:
        # --- Surgical Kill for the driver ---
        if driver:
            logging.info(f"[{name}] (Thread) Finished. Attempting to shut down its Selenium driver.")
            pid_to_kill = None
            try:
                pid_to_kill = driver.service.process.pid
            except Exception:
                pass
            
            try:
                driver.quit()
                logging.info(f"[{name}] (Thread) driver.quit() successful.")
            except Exception as e:
                logging.warning(f"[{name}] (Thread) driver.quit() failed: {e}. Attempting surgical kill.")
                if pid_to_kill:
                    try:
                        os.kill(pid_to_kill, 9)
                        logging.info(f"[{name}] (Thread) Successfully killed stuck driver process PID {pid_to_kill}.")
                    except Exception as e_kill:
                        logging.error(f"[{name}] (Thread) Failed to kill process PID {pid_to_kill}: {e_kill}")
                else:
                    logging.error(f"[{name}] (Thread) driver.quit() failed, but PID was not found. A zombie process may remain.")
            

# --- REFACTORED: scrape_all() ---
def scrape_all():
    """Runs all scraping jobs defined in SOURCE_CONFIG in parallel."""
    logging.info("--- Starting new scraping job (Parallel Mode) ---")
    
    session = create_robust_session()
    
    proxies_dict = None
    if PROXY_SETTINGS["use_proxies"] and PROXY_SETTINGS["proxy_url"]:
        logging.info(f"Proxy is ENABLED. Routing requests through: {PROXY_SETTINGS['proxy_url']}")
        proxies_dict = {
            "http": PROXY_SETTINGS["proxy_url"],
            "https": PROXY_SETTINGS["proxy_url"]
        }
    else:
        logging.info("Proxy is DISABLED.")
    
    all_counts = {}
    total_saved = 0
    futures = []
    
    executor = ThreadPoolExecutor(max_workers=len(SOURCE_CONFIG))

    try:
        # 1. Submit all jobs
        for source in SOURCE_CONFIG:
            future = executor.submit(scrape_source_wrapper, source, session, proxies_dict)
            futures.append(future)

        logging.info(f"Submitted {len(futures)} jobs to thread pool. Waiting up to 300s for completion...")
        
        # 2. Wait for jobs to complete, with a 5-minute (300s) timeout
        done, not_done = wait(futures, timeout=300)

        # 3. Process completed jobs
        for future in done:
            try:
                name, count = future.result()
                all_counts[name] = count
                total_saved += count
            except Exception as e:
                logging.error(f"A future job resulted in an error: {e}")
        
        # 4. Handle jobs that timed out
        if not_done:
            logging.critical(f"--- TIMEOUT: {len(not_done)} scrape jobs did not complete in 300s. ---")
            for future in not_done:
                logging.error("A thread has timed out and will be abandoned.")
                all_counts["Timed_Out_Jobs"] = all_counts.get("Timed_Out_Jobs", 0) + 1

    except Exception as e:
        logging.critical(f"--- CRITICAL: The entire scrape_all job failed. --- {e}")
        
    finally:
        # 5. Shut down the executor
        logging.info("Shutting down thread pool (wait=False)...")
        executor.shutdown(wait=False)
        
        # Create a dynamic log message
        log_summary = ", ".join(f"{count} {name}" for name, count in all_counts.items())
        log_message = f"Scraped: {log_summary} articles. (Total saved: {total_saved})"
        
        logging.info(log_message)
        print(log_message)
        
        logging.info("--- Scraping job finished ---")


# --- main() function with cleanup ---
def main():
    """
    Main function to run the scraper once.
    Includes robust error handling and DB connection closing.
    """
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
        
        # Force process exit to kill zombie threads (especially for Selenium)
        logging.info("--- Main thread finished. Forcing process exit to kill zombie threads. ---")
        os._exit(0)

if __name__ == '__main__':
    main()
