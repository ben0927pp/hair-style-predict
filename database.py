import os
import time
import urllib.request
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import random

# 建立資料夾存放圖片
os.makedirs('hairstyle_images', exist_ok=True)

# 隨機 User-Agent 用來防止被封鎖
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edge/91.0.864.59",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
]

# 啟動 Selenium
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 無頭模式
options.add_argument(f"user-agent={random.choice(user_agents)}")  # 隨機選擇一個 User-Agent
driver = webdriver.Chrome(options=options)

# 打開小紅書並搜尋「髮型」
url = 'https://www.xiaohongshu.com/'
driver.get(url)
time.sleep(3)

# 等待搜尋框加載並輸入搜尋關鍵字
search_box = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="text"]'))
)
search_box.send_keys('髮型')
search_box.send_keys(Keys.ENTER)
time.sleep(5)

# 爬取圖片和標題
data = []
soup = BeautifulSoup(driver.page_source, 'html.parser')
notes = soup.find_all('div', class_='note-item')  # 調整 class 根據實際網頁

for idx, note in enumerate(notes):
    try:
        title = note.find('h3').text.strip()
        img_tag = note.find('img')
        img_url = img_tag['src'] if img_tag else None

        if img_url:
            # 檢查並處理相對 URL
            if img_url.startswith('/'):
                img_url = f'https://www.xiaohongshu.com{img_url}'

            img_path = f'hairstyle_images/hairstyle_{idx}.jpg'
            urllib.request.urlretrieve(img_url, img_path)
            data.append({'Title': title, 'Image_Path': img_path})
            print(f"已下載: {img_path}")
    except Exception as e:
        print(f"錯誤: {e}")
        continue

# 儲存到 CSV
df = pd.DataFrame(data)
df.to_csv('hairstyle_data.csv', index=False, encoding='utf-8')
print('所有資料已保存到 hairstyle_data.csv')

driver.quit()