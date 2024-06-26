import urllib.request
from bs4 import BeautifulSoup
import requests
import json
import random
import time
import pandas as pd
import logging
import os
import numpy as np
import re
from tqdm import tqdm

class SpyderProcess:
    def __init__(self, headers, proxies, tencent_url, hainan_url,tecent_out_path, hainan_out_path,final_output_path):
        self.headers = headers
        self.proxies = proxies
        self.tencent_url = tencent_url
        self.hainan_url = hainan_url
        self.tecent_out_path = tecent_out_path
        self.hainan_out_path = hainan_out_path
        self.final_output_path = final_output_path

# --------------------------------腾讯云千帆产品爬取-------------------------------------- #
    def tencent_spyder(self):
        """
        腾讯云千帆产品市场的网页结构较为复杂，需要多层爬取，
        并且进行正则化处理以抽取所需数据，构造所需要的虚拟变量
        """
        proxies = self.proxies
        tencent_url = self.tencent_url

        productid = []
        name = []
        desc = []
        tags = []
        price = []

        for num in tqdm(range(10),desc='tencent pages'):
            payload = {
                'Action': 'SearchProducts',
                'CapiAuthErrorContinue': '1',
                'User': '0',
                'Version': '2019-10-10',
                'data': {
                    'DeviceId': 'XpycpEPAT7HajFaiJPwSdmEDeC3G0P99',
                    'Offset': 2 * num,
                    'Limit': 2,
                    'Source': '1'}
            }
            pro_json = requests.post(tencent_url, headers=self.headers, json=payload, proxies=proxies)
            pro_json.encoding = 'utf-8'
            pro_list = pro_json.json()['Response']['ProductSet']

            for i in pro_list:
                productid.append(i['ProductId'])
                name.append(i['ProductName'])
                desc.append(i['Summary'])
                tags.append("，".join(i['Tags']))
                price.append(i['MinPrice']['Price'] / 100)
            time.sleep(random.random())

        print('Tencent ProductId collection complete!')

        cicle = []
        spec = []

        for m in tqdm(productid, total=len(productid), desc='tencent productid'):
            new_url = f'https://market.cloud.tencent.com/products/{m}'
            req = urllib.request.Request(new_url, headers=self.headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
            response = opener.open(req).read().decode('utf-8')
            soup = BeautifulSoup(response, 'lxml')
            text1 = soup.select('a.mp__tag-switch-item:not(.mp__tag-switch-item-actived)')
            text2 = soup.find_all('a', class_='mp__tag-switch-item mp__tag-switch-item-actived')
            try:
                new_spec = text2[0]['value'] if not text1 else text1[0]['value']
                new_cicle = text2[1]['value'] if text2 else '-1'
            except:
                new_spec = '-1'
                new_cicle = '-1'
            spec.append(new_spec)
            cicle.append(new_cicle)

        df = pd.DataFrame({
            'name': name,
            'desc': desc,
            'tags': tags,
            'price': price,
            'cicle': cicle,
            'spec': spec
        })

        pattern = r'^([\d\.]+)元/([\d,]+)次$'
        match = df['spec'].str.extract(pattern, expand=False)
        price = match[0].str.replace(',', '').astype(float)
        times = match[1].str.replace(',', '').fillna(0).astype(int)
        mask = match.notnull().all(axis=1)
        df['new_spec'] = np.where(mask, (price / times).round(6).astype(str), df['spec'])

        df['pay_type'] = np.nan
        pay_type_0 = df['spec'].notna() & df['spec'].str.match(r'^[\d\.]+元/[\d,]+次$')
        pay_type_2 = df['cicle'].notna() & df['cicle'].str.contains(r'年|月|天')
        pay_type_1 = df['cicle'] == '1次'
        df.loc[pay_type_0, 'pay_type'] = 0
        df.loc[pay_type_2 & ~pay_type_0, 'pay_type'] = 2
        df.loc[pay_type_1, 'pay_type'] = 1

        df.loc[df['pay_type'] == 0, 'price'] = df['new_spec']
        
        for i, row in df[df['pay_type'] == 2].iterrows():
            cicle = row['cicle']
            if '天' in cicle:
                days = int(re.findall(r'\d+', cicle)[0])
                df.at[i, 'price'] = row['price'] * 365 / days
            elif '月' in cicle:
                months = int(re.findall(r'\d+', cicle)[0])
                df.at[i, 'price'] = row['price'] * 12 / months
            elif '年' in cicle:
                years = int(re.findall(r'\d+', cicle)[0])
                df.at[i, 'price'] = row['price'] / years

        df.to_csv(self.tecent_out_path, sep=',', index=False, header=True, encoding='utf_8_sig')
        
        return df

# --------------------------------海南数据产品市场爬取-------------------------------------- #
    def hainan_spyder(self):
        """
        海南数据产品超市的数据指标较为齐全，网页结构简单，但需要跟腾讯云千帆产品市场的处理指标保持一致
        """
        proxies = self.proxies
        hainan_url = self.hainan_url

        results = {
            'name': [],
            'desc': [],
            'tags':[],
            'price': [],
            'cicle':[],
            'spec':[],
            'new_spec':[],
            'pay_type': []
        }

        for pageNo in tqdm(range(1, 3),desc='hainan pages'):
            data = json.dumps({
                'pageNo': pageNo,
                'pageSize': 12,
                'searchKey': '',
                'timestamp': ''
            })
            response = requests.post(hainan_url, data=data, headers=self.headers, proxies=proxies)
            response.encoding = 'utf-8'
            pro_list = response.json()['data']['list']

            for item in pro_list:
                results['name'].append(item['proResourceName'])
                results['desc'].append(item['proResourceDesc'])
                results['tags'].append("")
                results['price'].append(item['itemPrice'])
                results['cicle'].append("")
                results['spec'].append("")
                results['new_spec'].append("")
                results['pay_type'].append(item['applyCount'])

            time.sleep(random.random() * 5)

        df2 = pd.DataFrame(results)
        print('Hainan spider is complete!')
        df2.to_csv(self.hainan_out_path, sep=',', index=False, header=True, encoding='utf_8_sig')
        return df2
    
    def combine_and_save(self, df, df2):
        final_df = pd.concat([df, df2], axis=0, ignore_index=True, sort=False)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.final_output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        final_df.to_csv(self.final_output_path, sep=',', index=False, header=True, encoding='utf_8_sig')
        print(f'Data saved to {self.final_output_path}')

    def run(self):
        try:
            df = self.tencent_spyder()
            df2 = self.hainan_spyder()
            self.combine_and_save(df, df2)
        except Exception as e:
            logging.basicConfig(filename='spider.log', level=logging.ERROR, 
                                format='%(asctime)s - %(levelname)s - %(message)s')
            logging.error(f"An error occurred while running the spider: {e}")
            print(f"An error occurred while running the spider: {e}")


if __name__ == '__main__':
    
    headers = {'*********************'} # 请替换为你的headers
    proxies = {"*********************"} # 请替换为你的IP代理
    
    tencent_url = 'https://market.cloud.tencent.com/ncgi/capi'
    hainan_url = 'https://www.datadex.cn/api/resource/searchBy'
    tecent_out_path = 'data/tecent_data.csv'
    hainan_out_path = 'data/hainan_data.csv'
    final_output_path = 'data/ori_spyder_data.csv'

    # excute
    spyder = SpyderProcess(headers, proxies, tencent_url, hainan_url, tecent_out_path,hainan_out_path,final_output_path)
    spyder.run()
