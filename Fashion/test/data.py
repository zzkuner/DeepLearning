import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_images(url, folder, session):
    response = session.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')

    for img in images:
        img_url = img.get('src')
        if not img_url:
            continue

        full_url = urljoin(url, img_url)
        try:
            img_data = session.get(full_url).content
            file_name = os.path.join(folder, img_url.split('/')[-1])
            with open(file_name, 'wb') as file:
                 file.write(img_data)
        except Exception as e:
            print(f"Failed to download {full_url}: {e}")

# 设置会话
session = requests.Session()

# 创建保存图片的文件夹
categories = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for category in categories:
    os.makedirs(f'./data/{category}', exist_ok=True)
    url = f'https://s.taobao.com/search?q={category}'
    download_images(url, f'./data/{category}', session)