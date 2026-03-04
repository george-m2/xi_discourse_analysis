import os
import csv
import time
import random
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup

# ---------------------------------------------------------
# GLOBAL CONFIGURATION & STEALTH SETTINGS
# ---------------------------------------------------------
TOTAL_PAGES_TO_SCRAPE = 1486 
OUTPUT_FILE = "master_xi_speeches.csv"
PHASE1_CACHE = "phase1_checkpoint.json"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/114.0.1823.58'
]

# Reusing the TCP connection makes scraping faster and kinder to the server
session = requests.Session()


# ---------------------------------------------------------
# PHASE 1: THE API LINK HARVESTER
# ---------------------------------------------------------
def harvest_links_via_api(max_pages):
    print(f"\n--- Phase 1: Harvesting API Links (Target: {max_pages} pages) ---")
    api_url = "https://jhsjk.people.cn/testnew/result"
    
    # 1. LOAD THE CACHE
    if os.path.exists(PHASE1_CACHE):
        try:
            with open(PHASE1_CACHE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            start_page = state.get('last_page', 0) + 1
            all_article_links = state.get('links', [])
            print(f"[Cache Found] Resuming Phase 1 from Page {start_page}. Holding {len(all_article_links)} links.")
        except Exception as e:
            print(f"[Cache Error] Could not read {PHASE1_CACHE}: {e}. Starting fresh.")
            start_page = 1
            all_article_links = []
    else:
        start_page = 1
        all_article_links = []
        
    if start_page > max_pages:
        print("Phase 1 is already 100% complete based on the cache!")
        return list(set(all_article_links))
        
    start_time = time.time() 
    
    # 2. START HARVESTING
    for page in range(start_page, max_pages + 1):
        params = {
            'keywords': '', 'isFuzzy': '0', 'searchArea': '0', 'year': '0',
            'form': '0', 'type': '0', 'page': page, 'sortType': '2', 
            'origin': '全部', 'source': '2'
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stealth_headers = {
                    'User-Agent': random.choice(USER_AGENTS),
                    'Referer': 'https://jhsjk.people.cn/result'
                }
                
                response = session.get(api_url, params=params, headers=stealth_headers, timeout=15)
                
                # Check for 403 Forbidden or 502 Bad Gateway
                if response.status_code != 200:
                    print(f"[Warning] Server returned {response.status_code}. Retrying in 15 seconds...")
                    time.sleep(15)
                    continue
                
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    text_content = response.text.strip("'\" \n\r")
                    if not text_content: 
                        raise ValueError("Server returned a completely blank response.")
                    data = json.loads(text_content)

                if data.get('status') == 'success':
                    article_list = data.get('list', [])
                    
                    if not article_list:
                        print("\nReached the absolute end of the database.")
                        return list(set(all_article_links))
                        
                    for article in article_list:
                        article_id = article.get('article_id')
                        if article_id:
                            all_article_links.append(f"https://jhsjk.people.cn/article/{article_id}")
                    
                    # 3. SAVE THE CACHE
                    with open(PHASE1_CACHE, 'w', encoding='utf-8') as f:
                        json.dump({'last_page': page, 'links': all_article_links}, f)
                    
                    # 4. ETA & STEALTH DELAYS
                    elapsed_time = time.time() - start_time
                    pages_processed = page - start_page + 1
                    avg_time_per_page = elapsed_time / pages_processed
                    eta_seconds = int(avg_time_per_page * (max_pages - page))
                    eta_mins, eta_secs = divmod(eta_seconds, 60)
                    
                    print(f"Page {page}/{max_pages} | Queued: {len(all_article_links)} links | ETA: {eta_mins}m {eta_secs}s")
                    
                    time.sleep(random.uniform(2.5, 4.5))
                    
                    if page % 30 == 0:
                        break_time = random.uniform(60, 120)
                        print(f"\n[Stealth] Taking a {int(break_time)} second coffee break to fool the firewall...")
                        time.sleep(break_time)
                        
                    break # Success! Move to next page
                
                else:
                    print(f"API returned an unexpected status. Retrying in 15 seconds...")
                    time.sleep(15)
                    
            except Exception as e:
                print(f"[Attempt {attempt + 1}/{max_retries}] Failed on page {page}: {e}")
                if attempt < max_retries - 1:
                    print("Sleeping for 15 seconds for the firewall to reset...")
                    time.sleep(15)
                else:
                    print(f"CRITICAL: Failed to get page {page} after {max_retries} attempts.")
                    return list(set(all_article_links)) # Return what we have so far
            
    return list(set(all_article_links))


# ---------------------------------------------------------
# PHASE 2: THE CONTENT EXTRACTOR
# ---------------------------------------------------------
def extract_speech_text(url):
    stealth_headers = {
        'User-Agent': random.choice(USER_AGENTS)
    }
    try:
        response = session.get(url, headers=stealth_headers, timeout=15)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Title
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"
        
        # 2. Date
        date_divs = soup.find_all('div', class_='d2txt_1 clearfix')
        publication_date = "Unknown Date"
        for div in date_divs:
            text = div.get_text(strip=True)
            if '发布时间' in text:
                publication_date = text.split('发布时间：')[-1].strip()
                
        # 3. Text
        text_content = []
        article_body = soup.find('div', class_='d2txt_con clearfix') 
        if article_body:
            for p in article_body.find_all('p'):
                text = p.get_text(strip=True)
                if text: 
                    text_content.append(text)
                    
        full_text = "\n".join(text_content)
        
        return {
            'publication': publication_date,
            'title': title,
            'url': url,
            'text': full_text
        }
    except Exception as e:
        print(f"\n[Error Extracting] {url}: {e}")
        return None


# ---------------------------------------------------------
# EXECUTION ENGINE
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Step 1: Harvest URLs
    all_urls = harvest_links_via_api(max_pages=TOTAL_PAGES_TO_SCRAPE)
    
    # Step 2: Phase 2 Checkpointing (Resume from CSV)
    scraped_urls = set()
    file_exists = os.path.exists(OUTPUT_FILE)
    
    if file_exists:
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            if 'url' in df_existing.columns:
                scraped_urls = set(df_existing['url'].tolist())
                print(f"\n[Checkpoint Found] Discovered existing CSV. Skipping {len(scraped_urls)} previously downloaded speeches.")
        except Exception as e:
            print(f"\n[Warning] Could not read existing CSV for checkpointing: {e}")
            
    urls_to_scrape = [url for url in all_urls if url not in scraped_urls]
    total_remaining = len(urls_to_scrape)
    
    if total_remaining == 0:
        print("\nAll URLs have already been scraped! The dataset is complete.")
        exit()
        
    print(f"\n--- Phase 2: Extracting {total_remaining} REMAINING speeches ---")
    
    start_time_phase2 = time.time()
    
    # Step 3: Extract and Save Instantly
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['publication', 'title', 'url', 'text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:
            writer.writeheader()
            
        for index, url in enumerate(urls_to_scrape):
            data = extract_speech_text(url)
            
            if data and data['text']:
                writer.writerow(data)
                f.flush() # Securely save to disk
            
            # Phase 2 ETA Math
            current_item = index + 1
            elapsed_time = time.time() - start_time_phase2
            avg_time_per_url = elapsed_time / current_item
            eta_seconds = int(avg_time_per_url * (total_remaining - current_item))
            eta_hours, remainder = divmod(eta_seconds, 3600)
            eta_mins, eta_secs = divmod(remainder, 60)
            
            time_str = f"{eta_hours}h {eta_mins}m {eta_secs}s" if eta_hours > 0 else f"{eta_mins}m {eta_secs}s"
            
            print(f"[{current_item}/{total_remaining}] Saved {url.split('/')[-1]} -> ETA: {time_str}")
                
            time.sleep(random.uniform(1.5, 3.5)) 

    print(f"\nSuccess! All remaining {total_remaining} speeches downloaded to {OUTPUT_FILE}.")