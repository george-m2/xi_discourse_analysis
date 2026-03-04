import os
import csv
import time
import random
import requests
import pandas as pd
import json
from bs4 import BeautifulSoup

# Initialize a global session for network efficiency (reusing the TCP connection)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://jhsjk.people.cn/result'
})

# ---------------------------------------------------------
# PHASE 1: THE API LINK HARVESTER
# ---------------------------------------------------------
def harvest_links_via_api(max_pages=1486):
    print(f"\n--- Phase 1: Harvesting API Links (Target: {max_pages} pages) ---")
    api_url = "https://jhsjk.people.cn/testnew/result"
    all_article_links = []
    start_time = time.time() 
    
    for page in range(1, max_pages + 1):
        params = {
            'keywords': '', 'isFuzzy': '0', 'searchArea': '0', 'year': '0',
            'form': '0', 'type': '0', 'page': page, 'sortType': '2', 
            'origin': '全部', 'source': '2'
        }
        
        try:
            response = session.get(api_url, params=params, timeout=15)
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = json.loads(response.text.strip("'\""))

            if data.get('status') == 'success':
                article_list = data.get('list', [])
                
                if not article_list:
                    print("\nReached the absolute end of the database.")
                    break
                    
                for article in article_list:
                    article_id = article.get('article_id')
                    if article_id:
                        all_article_links.append(f"https://jhsjk.people.cn/article/{article_id}")
                
                # ETA Math
                elapsed_time = time.time() - start_time
                avg_time_per_page = elapsed_time / page
                eta_seconds = int(avg_time_per_page * (max_pages - page))
                eta_mins, eta_secs = divmod(eta_seconds, 60)
                
                print(f"Page {page}/{max_pages} | Queued: {len(all_article_links)} links | ETA: {eta_mins}m {eta_secs}s")
                time.sleep(random.uniform(1.0, 2.5))
            else:
                print("API returned an unexpected response.")
                break
                
        except Exception as e:
            print(f"Failed to reach API on page {page}: {e}")
            break
            
    # Remove duplicates
    return list(set(all_article_links))

# ---------------------------------------------------------
# PHASE 2: THE CONTENT EXTRACTOR
# ---------------------------------------------------------
def extract_speech_text(url):
    try:
        response = session.get(url, timeout=15)
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
# EXECUTION (WITH CHECKPOINTING)
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Configuration
    TOTAL_PAGES_TO_SCRAPE = 1486 
    OUTPUT_FILE = "master_xi_speeches.csv"
    
    # Step 1: Harvest all links from the API
    all_urls = harvest_links_via_api(max_pages=TOTAL_PAGES_TO_SCRAPE)
    
    # Step 2: Implement Checkpointing (Resume capability)
    scraped_urls = set()
    file_exists = os.path.exists(OUTPUT_FILE)
    
    if file_exists:
        try:
            # Read the existing CSV to see what we already downloaded
            df_existing = pd.read_csv(OUTPUT_FILE)
            if 'url' in df_existing.columns:
                scraped_urls = set(df_existing['url'].tolist())
                print(f"\n[Checkpoint Found] Discovered existing CSV. Skipping {len(scraped_urls)} previously downloaded speeches.")
        except Exception as e:
            print(f"\n[Warning] Could not read existing CSV for checkpointing: {e}")
            
    # Filter out the URLs we already have
    urls_to_scrape = [url for url in all_urls if url not in scraped_urls]
    total_remaining = len(urls_to_scrape)
    
    if total_remaining == 0:
        print("\nAll URLs have already been scraped! The dataset is complete.")
        exit()
        
    print(f"\n--- Phase 2: Extracting {total_remaining} REMAINING speeches ---")
    
    start_time_phase2 = time.time()
    
    # Step 3: Open file in 'append' mode and write line-by-line
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['publication', 'title', 'url', 'text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write the header row only if the file is brand new
        if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:
            writer.writeheader()
            
        for index, url in enumerate(urls_to_scrape):
            data = extract_speech_text(url)
            
            # If extraction is successful and contains actual text, save it immediately
            if data and data['text']:
                writer.writerow(data)
                f.flush() # Force OS to save to disk right now to prevent data loss
            
            # ETA Math for Phase 2
            current_item = index + 1
            elapsed_time = time.time() - start_time_phase2
            avg_time_per_url = elapsed_time / current_item
            eta_seconds = int(avg_time_per_url * (total_remaining - current_item))
            eta_hours, remainder = divmod(eta_seconds, 3600)
            eta_mins, eta_secs = divmod(remainder, 60)
            
            time_str = f"{eta_hours}h {eta_mins}m {eta_secs}s" if eta_hours > 0 else f"{eta_mins}m {eta_secs}s"
            
            print(f"[{current_item}/{total_remaining}] Saved {url.split('/')[-1]} -> ETA: {time_str}")
                
            # Polite crawler sleep
            time.sleep(random.uniform(1.5, 3.5)) 

    print(f"\nSuccess! All remaining {total_remaining} speeches downloaded to {OUTPUT_FILE}.")