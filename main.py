import requests
import time
import random
import pandas as pd
import json
from bs4 import BeautifulSoup

# ---------------------------------------------------------
# PHASE 1: THE API LINK HARVESTER
# ---------------------------------------------------------
def harvest_links_via_api(max_pages=5):
    print(f"--- Starting Phase 1: Harvesting API Links ({max_pages} pages) ---")
    api_url = "https://jhsjk.people.cn/testnew/result"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://jhsjk.people.cn/result'
    }
    
    all_article_links = []
    start_time = time.time() # Start the timer
    
    for page in range(1, max_pages + 1):
        # These are the exact parameters the website uses to filter the database
        params = {
            'keywords': '', 'isFuzzy': '0', 'searchArea': '0', 'year': '0',
            'form': '0', 'type': '0', 'page': page, 'sortType': '2', 
            'origin': '全部', 'source': '2'
        }
        
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            
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
                
                # --- ETA MATH FOR PHASE 1 ---
                elapsed_time = time.time() - start_time
                avg_time_per_page = elapsed_time / page
                pages_left = max_pages - page
                eta_seconds = int(avg_time_per_page * pages_left)
                eta_mins, eta_secs = divmod(eta_seconds, 60)
                
                print(f"Page {page}/{max_pages} | Found: {len(article_list)} links | Total queued: {len(all_article_links)} | ETA: {eta_mins}m {eta_secs}s")
                
                # BE POLITE: Wait 1 to 2 seconds before asking for the next page
                time.sleep(random.uniform(1.0, 2.5))
            else:
                print("API returned an unexpected response.")
                break
                
        except Exception as e:
            print(f"Failed to reach API on page {page}: {e}")
            break
            
    return list(set(all_article_links))


# ---------------------------------------------------------
# PHASE 2: THE CONTENT EXTRACTOR
# ---------------------------------------------------------
def extract_speech_text(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"
        
        date_divs = soup.find_all('div', class_='d2txt_1 clearfix')
        publication_date = "Unknown Date"
        for div in date_divs:
            text = div.get_text(strip=True)
            if '发布时间' in text:
                publication_date = text.split('发布时间：')[-1].strip()
                
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
        return None

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. SET YOUR TARGET PAGES HERE (e.g., 1486 for the whole database)
    TOTAL_PAGES_TO_SCRAPE = 10 
    
    urls_to_scrape = harvest_links_via_api(max_pages=TOTAL_PAGES_TO_SCRAPE)
    
    # 2. Extract Text
    total_urls = len(urls_to_scrape)
    print(f"\n--- Starting Phase 2: Extracting text from {total_urls} URLs ---")
    dataset = []
    
    start_time_phase2 = time.time() # Start Phase 2 timer
    
    for index, url in enumerate(urls_to_scrape):
        data = extract_speech_text(url)
        if data and data['text']:
            dataset.append(data)
            
        # --- ETA MATH FOR PHASE 2 ---
        current_item = index + 1
        elapsed_time = time.time() - start_time_phase2
        avg_time_per_url = elapsed_time / current_item
        urls_left = total_urls - current_item
        eta_seconds = int(avg_time_per_url * urls_left)
        
        # Convert total seconds into hours, minutes, and seconds for readability
        eta_hours, remainder = divmod(eta_seconds, 3600)
        eta_mins, eta_secs = divmod(remainder, 60)
        
        # Format the time string conditionally
        if eta_hours > 0:
            time_str = f"{eta_hours}h {eta_mins}m {eta_secs}s"
        else:
            time_str = f"{eta_mins}m {eta_secs}s"
            
        print(f"[{current_item}/{total_urls}] Downloaded. ETA: {time_str}")
            
        # Hard sleep so your IP doesn't get blocked
        time.sleep(random.uniform(1.5, 3.5)) 

    # 3. Save to CSV
    df = pd.DataFrame(dataset)
    df.to_csv("master_xi_speeches.csv", index=False, encoding='utf-8-sig')
    print(f"\nSuccess! Downloaded {len(df)} speeches to master_xi_speeches.csv")