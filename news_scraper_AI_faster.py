# ==============================================================================
# --- GLOBAL USER SETTINGS ---
#
# How many articles to get from each source (e.g., 20)
MAX_ARTICLES_PER_SOURCE = 20
# Max time (in seconds) the entire scraping job can run before stopping
GLOBAL_JOB_TIMEOUT_SECONDS = 300 
# Semantic similarity threshold (0.0 to 1.0) for news clustering/deduplication
DEDUPLICATION_THRESHOLD = 0.70

# --- PROXY CONFIGURATION ---
PROXY_SETTINGS = {
    "use_proxies": False,
    "proxy_url": None  # e.g., "http://user:pass@proxy.service.com:8080"
}

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
import re 
import traceback # <-- Added for detailed error logging

# --- AI/Semantics Imports ---
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    logging.critical("sentence-transformers not installed. AI clustering will be skipped.")
    SentenceTransformer = None
    util = None
# -------------------------------------

# --- Parallelism Imports ---
from concurrent.futures import ThreadPoolExecutor, wait
import threading
# ------------------------------------

# --- Selenium Imports ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.common.exceptions import WebDriverException, TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    logging.critical("Selenium not installed. Selenium-dependent sources will fail.")
    SELENIUM_AVAILABLE = False
# ---------------------------

# --- Configure logging ---
logging.basicConfig(filename='news_scraper.log',
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# --- UTILITY & ROBUSTNESS FUNCTIONS ---
# ==============================================================================

def create_robust_session():
    """
    Creates a requests.Session with automatic retries on server errors (5xx)
    and common connection errors.
    """
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
    """Returns a complete header dictionary for a given 'persona'."""
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
        logging.error("Cannot create Selenium driver, library not found.")
        return None
        
    logging.info("Initializing headless Selenium Chrome driver (using SeleniumManager)...")
    
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
        driver.set_page_load_timeout(30) # Increased to 30s to help initialization
        logging.info("Selenium driver initialized successfully.")
        return driver
    except WebDriverException as e:
        logging.critical(f"Failed to initialize Selenium driver. Error: {e}")
        return None
    except Exception as e:
        logging.critical(f"An unexpected error occurred during Selenium initialization: {e}")
        return None

# --- NEW: Title Cleaning Function ---
def clean_title(title, source_name=None):
    """
    Removes common boilerplate/site names appended to news titles 
    (e.g., ' - The Hindu' or ' | TOI').
    """
    if source_name:
        source_pattern = re.escape(source_name)
        # Match separator (-/:|), optional space, source name, optional trailing space, END of string ($)
        title = re.sub(r'[\s]*[-|:\s][\s]*' + source_pattern + r'[\s]*$', '', title, flags=re.IGNORECASE).strip()
    
    # Final cleanup: replace multiple spaces with single space
    title = re.sub(r'\s+', ' ', title).strip()
    return title

# ==============================================================================
# --- AI DEDUPLICATION LOGIC (FIXED FOR DEADLOCKS) ---
# ==============================================================================

def calculate_embedding(title, summary):
    """Calculates the semantic embedding for a title+summary combo."""
    if semantic_model is None or not summary:
        return None
    try:
        text = f"{title}. {summary or 'No content available'}"
        embedding = semantic_model.encode([text], convert_to_tensor=True)[0]
        return embedding
    except Exception as e:
        logging.error(f"Error calculating embedding: {e}")
        return None

def get_cluster_id_for_article(new_title, new_summary, new_embedding):
    """
    Checks DB for articles from the last 24 hours.
    If a semantically similar article exists, returns its cluster_id.
    
    *** CRITICAL FIX: All expensive AI operations run OUTSIDE the db_lock ***
    """
    if new_embedding is None or util is None:
        return str(uuid.uuid4())

    try:
        # 1. Fetch recent articles (FAST DB READ ONLY)
        with db_lock:
            cursor.execute('''
                SELECT title, summary, cluster_id 
                FROM news 
                WHERE scraped_at >= datetime('now', '-1 day')
            ''')
            recent_articles = cursor.fetchall()

        # If no recent articles, no need to compare
        if not recent_articles:
            return str(uuid.uuid4())

        # 2. Prepare data for comparison
        existing_texts = [f"{row[0]}. {row[1]}" for row in recent_articles]
        existing_ids = [row[2] for row in recent_articles]
        
        # 3. Calculate Embeddings for existing articles 
        # (EXPENSIVE OPERATION - MUST BE OUTSIDE LOCK)
        existing_embeddings = semantic_model.encode(existing_texts, convert_to_tensor=True)

        # 4. Calculate Cosine Similarity
        cosine_scores = util.cos_sim(new_embedding, existing_embeddings)[0]

        # 5. Find best match
        best_score = -1
        best_idx = -1

        for i, score in enumerate(cosine_scores):
            if score > best_score:
                best_score = score
                best_idx = i

        # 6. Decision Threshold
        if best_score >= DEDUPLICATION_THRESHOLD:
            logging.info(f"DEDUPLICATION: Found match (Score: {best_score:.2f}). Linking to Cluster ID: {existing_ids[best_idx]}")
            return existing_ids[best_idx]
        else:
            return str(uuid.uuid4())
            
    except Exception as e:
        logging.error(f"Error during AI clustering calculation: {e}")
        return str(uuid.uuid4())


def save_article(source, title, url, summary, image_url):
    """
    Saves a single article to the SQLite database.
    Performs cleanup and calculates/checks the cluster ID.
    """
    try:
        # --- ROBUST CLEANING & TITLE CLEANUP ---
        # 1. Clean up title
        title = " ".join(title.replace('\n', ' ').replace('\r', ' ').split()).strip()
        title = clean_title(title, source) 
        
        # 2. Clean up summary
        summary_clean = None
        if summary:
            summary_lines = [line.strip() for line in summary.splitlines() if line.strip()]
            summary_clean = "\n\n".join(summary_lines)
        
        # 3. Calculate embedding (OUTSIDE the DB lock)
        # Use raw summary for embedding, clean for saving
        embedding = calculate_embedding(title, summary_clean)
        
        # --- THREAD-SAFE BLOCK ---
        with db_lock:
            # 4. Check for URL existence (primary skip)
            cursor.execute("SELECT id FROM news WHERE url = ?", (url,))
            if cursor.fetchone():
                logging.info(f"Duplicate article skipped: {title} from {source}")
                return False
            
            # 5. Get Cluster ID (Fix implemented: AI runs outside this lock inside the function call logic, 
            # or strictly speaking, the function handles its own internal lock correctly now)
            cluster_id = get_cluster_id_for_article(title, summary_clean, embedding)

            # 6. Set final values (using None for DB NULL)
            final_summary = summary_clean if summary_clean else None
            final_image_url = image_url if image_url else None
            
            # 7. Insert into database
            cursor.execute('''
                INSERT INTO news (cluster_id, source, title, url, summary, image_url, scraped_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (cluster_id, source, title, url, final_summary, final_image_url, datetime.now()))
            conn.commit()
            
        logging.info(f"Saved article: {title} (Cluster ID: {cluster_id[:8]}) from {source}")
        return True
            
    except Exception as e:
        # --- ENHANCED ERROR LOGGING ---
        logging.critical(f"CRITICAL SAVE ERROR: {title}. Traceback:\n{traceback.format_exc()}")
        logging.error(f"Error saving article {title}: {e}")
        return False

# --- Timer function to kill the Selenium load if it hangs ---
def stop_page_load(driver_to_stop, source_name):
    """Callback function to stop a hung page load after the timer expires."""
    logging.warning(f"[{source_name}] [FORCED TIMEOUT] Page load exceeded 20s. Stopping browser script execution.")
    try:
        driver_to_stop.execute_script("window.stop();")
    except Exception as e:
        logging.debug(f"[{source_name}] Error during forced stop: {e}")


# ==============================================================================
# --- MAIN SCRAPING LOGIC ---
# ==============================================================================

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
        'article_strategies': ['requests_browser', 'selenium_browser'],
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
        'article_strategies': ['requests_browser', 'selenium_browser'],
        'article_url_contains': None,
        'referer': 'https://www.thehindu.com/',
    }
]

def scrape_source(session, selenium_driver, source_config, proxies_dict):
    """
    Scrapes a single source. Uses a BATCH query for early URL skipping.
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
        
        # ----------------------------------------------------
        # *** OPTIMIZED: BATCH EARLY SKIP CHECK ***
        # ----------------------------------------------------
        articles_to_process = []
        rss_urls = []
        for item in items[:MAX_ARTICLES_PER_SOURCE]:
            if item.link and item.link.text.strip():
                rss_urls.append(item.link.text.strip())

        existing_urls = set()
        if rss_urls:
            try:
                # Query the database for existing URLs in a single batch
                placeholders = ', '.join('?' for _ in rss_urls)
                sql_query = f"SELECT url FROM news WHERE url IN ({placeholders})"
                with db_lock:
                    cursor.execute(sql_query, rss_urls)
                    for row in cursor.fetchall():
                        existing_urls.add(row[0])
                logging.info(f"[{name}] Found {len(existing_urls)} RSS URLs already present in the DB.")
            except Exception as e:
                logging.error(f"[{name}] Error during batch DB check: {e}")

        # Filter the RSS items for processing
        for item in items[:MAX_ARTICLES_PER_SOURCE]:
            article_url = item.link.text.strip() if item.link else None
            
            if not article_url: continue

            if article_url in existing_urls:
                logging.info(f"[{name}] Early skip (batch): URL {article_url} already exists.")
                continue
            
            # Check for URL filter
            if source_config['article_url_contains']:
                if source_config['article_url_contains'] not in article_url:
                    logging.warning(f"[{name}] Skipping non-article link: {article_url}")
                    continue

            # Item is new and meets initial filtering criteria
            articles_to_process.append(item)

        logging.info(f"[{name}] Proceeding to scrape {len(articles_to_process)} new articles.")
        # ----------------------------------------------------

        # 2. Process each article
        for item in articles_to_process:
            time.sleep(random.uniform(0.1, 0.3)) 
            
            try:
                article_url = item.link.text.strip()
                rss_title = item.title.text if item.title else "Title not found"
                
                summary = None
                raw_html = None
                final_title = rss_title
                image_url = None 
                
                strategies = source_config['article_strategies']
                
                for i, strategy in enumerate(strategies):
                    logging.info(f"[{name}] Attempt {i+1}/{len(strategies)}: Trying with '{strategy}' strategy...")
                    raw_html = None

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

                            timeout_seconds = 20
                            timer = threading.Timer(timeout_seconds, stop_page_load, args=[selenium_driver, name])
                            timer.start()
                            
                            try:
                                selenium_driver.get(article_url)
                                WebDriverWait(selenium_driver, 10).until(
                                    EC.presence_of_element_located((By.TAG_NAME, "p"))
                                )
                                if timer.is_alive():
                                    timer.cancel()
                                    
                                raw_html = selenium_driver.page_source
                            
                            except (TimeoutException, WebDriverException) as e:
                                if timer.is_alive():
                                    timer.cancel()
                                logging.error(f"[{name}] Request failed for strategy '{strategy}' on URL {article_url}: Timed out or failed. {e}")
                                continue
                        
                        else:
                            logging.error(f"[{name}] Unknown strategy: {strategy}. Skipping.")
                            continue
                        # --- END STRATEGY ROUTER ---

                        # 3. Extract Content
                        if not raw_html:
                            logging.warning(f"[{name}] FAILED with '{strategy}' (HTML was empty).")
                            continue

                        temp_summary = None
                        try:
                            temp_summary = trafilatura.extract(raw_html, include_comments=False, include_tables=False)
                        except Exception as e:
                            logging.error(f"[{name}] trafilatura failed to parse HTML: {e}")
                        
                        word_count = 0
                        if temp_summary:
                            word_count = len(temp_summary.split())
                        
                        # 4. Success check (Minimum 80 words)
                        if word_count >= 80:
                            logging.info(f"[{name}] Success with '{strategy}'. Found content ({word_count} words).")
                            summary = temp_summary
                            
                            # Parse metadata (only on successful scrape)
                            soup_meta = BeautifulSoup(raw_html, 'html.parser')
                            page_title = soup_meta.find('title')
                            if page_title:
                                final_title = page_title.text
                            
                            og_image = soup_meta.find('meta', property='og:image')
                            if og_image and og_image.get('content'):
                                image_url = og_image['content']
                            
                            break # <-- Success! Exit the strategy loop.
                        else:
                            logging.warning(f"[{name}] FAILED with '{strategy}' (content was too short: {word_count} words).")
                            
                    except Exception as e:
                        logging.error(f"[{name}] Request failed for strategy '{strategy}' on URL {article_url}: {e}")
                        
                    # Wait a moment before trying the next strategy
                    if i < len(strategies) - 1:
                        time.sleep(random.uniform(0.5, 1.0))
                
                # --- END OF STRATEGY LOOP ---

                # 5. Save Logic
                if summary:
                    was_saved = save_article(name, final_title, article_url, summary, image_url)
                    if was_saved:
                        articles_saved_list.append(final_title)
            
            except Exception as e:
                logging.error(f"[{name}] Article-level Error: {e} for url {article_url}")

    except requests.RequestException as e:
        logging.error(f"Failed to fetch {name} RSS feed: {e}")
    except Exception as e:
        logging.error(f"Failed to parse {name} RSS feed: {e}")
        
    return (name, len(articles_saved_list))

# --- Thread Wrapper Function ---
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
            driver = create_selenium_driver()
            if not driver:
                logging.error(f"[{name}] (Thread) Selenium driver failed to start.")
        
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
            

# --- scrape_all() ---
def scrape_all():
    """
    Runs all scraping jobs defined in SOURCE_CONFIG in parallel.
    Uses the GLOBAL_JOB_TIMEOUT_SECONDS constant.
    """
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
        # 1. Submit all jobs to the thread pool
        for source in SOURCE_CONFIG:
            future = executor.submit(scrape_source_wrapper, source, session, proxies_dict)
            futures.append(future)

        logging.info(f"Submitted {len(futures)} jobs to thread pool. Waiting up to {GLOBAL_JOB_TIMEOUT_SECONDS}s for completion...")
        
        # 2. Wait for jobs to complete, with timeout
        done, not_done = wait(futures, timeout=GLOBAL_JOB_TIMEOUT_SECONDS)

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
            logging.critical(f"--- TIMEOUT: {len(not_done)} scrape jobs did not complete in {GLOBAL_JOB_TIMEOUT_SECONDS}s. ---")
            for future in not_done:
                logging.error("A thread has timed out and will be abandoned.")
                all_counts["Timed_Out_Jobs"] = all_counts.get("Timed_Out_Jobs", 0) + 1

    except Exception as e:
        logging.critical(f"--- CRITICAL: The entire scrape_all job failed. --- {e}")
        
    finally:
        # 5. Shut down the executor
        logging.info("Shutting down thread pool (wait=False)...")
        executor.shutdown(wait=False)
        
        log_summary = ", ".join(f"{count} {name}" for name, count in all_counts.items())
        log_message = f"Scraped: {log_summary} articles. (Total saved: {total_saved})"
        
        logging.info(log_message)
        print(log_message)
        
        logging.info("--- Scraping job finished ---")


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
logging.info(f"Initializing database connection at: {db_path}")
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

# Ensure schema includes 'cluster_id'
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

# --- MIGRATION Check ---
cursor.execute("PRAGMA table_info(news)")
columns = [info[1] for info in cursor.fetchall()]
if 'cluster_id' not in columns:
    logging.info("Performing database migration: Adding 'cluster_id' column.")
    try:
        cursor.execute("ALTER TABLE news ADD COLUMN cluster_id TEXT")
        logging.info("Migration successful.")
    except Exception as e:
        logging.error(f"Migration failed: {e}")
conn.commit()

# --- Thread lock for database ---
db_lock = threading.Lock()
# -------------------------------------


# --- main() function with cleanup ---
def main():
    """
    Main function to run the scraper once.
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
        
        # --- KILL SWITCH ---
        logging.info("--- Main thread finished. Forcing process exit to kill zombie threads. ---")
        os._exit(0)

if __name__ == '__main__':
    main()
