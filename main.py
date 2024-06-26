import urllib.request
from bs4 import BeautifulSoup
import requests
import json
import random
import time
import logging
import os
import re
import heapq
import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from collections import Counter
from imblearn.over_sampling import SMOTE
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve, auc, silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tqdm import tqdm
from gensim.models import Word2Vec, HdpModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

from spyder import SpyderProcess
from cluster import Cluster
from textcnn import TextClassificationModel
from feature_reg import RegressionModel


# ===================爬取数据部分===================
headers = {'*********************'} # 请替换为你的headers
proxies = {"*********************"} # 请替换为你的IP代理

tencent_url = 'https://market.cloud.tencent.com/ncgi/capi'
hainan_url = 'https://www.datadex.cn/api/resource/searchBy'
tecent_out_path = 'data/tecent_data.csv'
hainan_out_path = 'data/hainan_data.csv'
final_output_path = 'data/ori_spyder_data.csv'
spyder = SpyderProcess(headers, proxies, tencent_url, hainan_url, tecent_out_path,hainan_out_path,final_output_path)
spyder.run()

# ===================聚类以及验证部分===================
input_data = 'data/ori_spyder_data.csv'
threshold = 0.3  # 相似度阈值
stopwords_file = './hit_stopwords.txt' 
output_data = './data/selected_data_with_label.csv'
best_cluster_num = 4
cluster = Cluster(input_data, threshold,stopwords_file,best_cluster_num, output_data)
cluster.cluster()
cluster.extract_topics()

# ===================TEXTCNN构建=======================
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
input_data = './data/selected_data_with_label.csv'
model = TextClassificationModel(input_data)
model.run()

# ===================基于tfidf特征提取的回归模型构建======
csv_file = './data/selected_data_with_label.csv'
output_file = './output/reg_model_summary.xlsx'
num_features = 20
model = RegressionModel(csv_file, num_features, output_file)
model.load_and_filter_data()
model.extract_tfidf_features()
model.forward_selected(method='aic')

