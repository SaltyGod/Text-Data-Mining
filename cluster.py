import pandas as pd
import jieba
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import Counter
import gensim
from gensim.models import HdpModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

class Cluster:
    def __init__(self, input_data, threshold,stopwords_file,best_cluster_num,output_data):
        self.input_data = input_data
        self.threshold = threshold
        self.stopwords_file = stopwords_file
        self.stopwords = self.load_stopwords()
        self.best_cluster_num = best_cluster_num
        self.output_data = output_data
        self.topic_output_file = 'output/group_{label}_topics.csv' 
    
    def load_stopwords(self):
        # 加载停用词表
        stopwords = set()
        with open(self.stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
        print('======================停用词表加载完成======================')
        return stopwords

    def clean_text(self, text):
        """
        清理文本，去除特殊字符、停用词和数字。
        """
        # 去除特殊字符
        text = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        # 分词
        seg_list = jieba.cut(text, cut_all=False)
        # 去除停用词
        words = [word for word in seg_list if word not in self.stopwords]
        # 连接词语
        text = ' '.join(words)
        # 去除数字
        text = re.sub(r"[0-9]+", "", text)
        return text

    def process(self):
        """
        处理输入数据并返回处理后的DataFrame。
        """
        df = pd.read_csv(self.input_data)
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['desc']) 
        similarity_matrix = cosine_similarity(X)

        similar_pairs = []
        num_deleted = 0
        for i in range(similarity_matrix.shape[0]):
            for j in range(i + 1, similarity_matrix.shape[0]):
                if similarity_matrix[i, j] >= self.threshold:
                    similar_pairs.append((i, j))
                    num_deleted += 1
        df.drop(df.index[[x[1] for x in similar_pairs]], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # 输出结果
        print(f"删除了{num_deleted}个样本，保留了{len(df)}个样本。")

        # 添加自定义词典
        words_to_add = ['小程序', '大数据', '大屏', '大会员', '大运营', '二要素', '框架', '时间段', '配送', '运送', '输送', '立式机', '闸机', '稳定性强', '线上', '线下', '派单', '不限流', '全流程']
        for word in words_to_add:
            jieba.add_word(word)
        
        return df

    def process_data(self):
        df = self.process()
        df['desc_process'] = df['desc'].apply(self.clean_text)
        print('======================数据清洗已完成======================')
        return df

    def outlier_removal(self, df):
        df = df.dropna(subset=['desc_process']).reset_index(drop=True) # 去除空值并重置索引
        # TF-IDF特征化
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['desc_process'].values.astype('U'))

        # Word2Vec向量化
        desc_process_cut = df['desc_process'].apply(lambda x: list(set(x.split())))
        model_w2v = Word2Vec(desc_process_cut, vector_size=150, window=5, min_count=1)
        
        w2v_vectors = []
        for words in desc_process_cut:
            vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
            if vectors:
                w2v_vectors.append(np.mean(vectors, axis=0))
            else:
                w2v_vectors.append(np.zeros(150))  # 使用零向量代替
        
        w2v_vectors = np.array(w2v_vectors)

        # LOF算法识别异常值
        lof = LocalOutlierFactor(n_neighbors=10)
        outlier_scores = lof.fit_predict(w2v_vectors)
        df['lof_label'] = outlier_scores

        # 剔除异常值
        cluster_counts = dict(Counter(outlier_scores))
        print(cluster_counts)
        print(f"======================异常值占比：{cluster_counts[-1] / len(df):.2%}，剔除异常值已完成======================")
        selected_data = df.loc[df['lof_label'] == 1]
        return selected_data

    def cluster(self):
        df = self.process_data()
        selected_data = self.outlier_removal(df)
        
        # TF-IDF特征化
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(selected_data['desc_process'].values.astype('U'))

        # 去重
        desc_process_cut = selected_data['desc_process'].apply(lambda x: list(set(x.split())))
        model_w2v = Word2Vec(desc_process_cut, vector_size=150, window=5, min_count=1)
        w2v_vectors = []
        for words in desc_process_cut:
            vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
            if vectors:
                w2v_vectors.append(np.mean(vectors, axis=0))
            else:
                w2v_vectors.append(np.zeros(150))  # 使用零向量代替

        w2v_vectors = np.array(w2v_vectors)

        # 计算轮廓系数并绘图
        silhouette_scores = []
        print('======================开始绘制轮廓系数图======================')
        for i in range(2, 11):
            score = self.calculate_silhouette(w2v_vectors, i)
            silhouette_scores.append(score)
            print(f"k={i}, silhouette score={score:.4f}")

        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette score')
        plt.show()

        # 计算肘部法并绘图
        wcss_scores = self.calculate_wcss(w2v_vectors)
        print('======================开始绘制WCSS图=====================')
        plt.plot(range(2, 11), wcss_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS score')
        plt.show()

        # 进行kmeans聚类
        print('======================基于最优聚类数开始聚类=====================')
        kmeans = KMeans(n_clusters=self.best_cluster_num, random_state=42)
        labels = kmeans.fit_predict(w2v_vectors)
        unique_labels = set(labels)
        label_counts = {label: list(labels).count(label) for label in unique_labels}
        print("每一类标签包含的样本数以及每一类样本对应的标签：", label_counts)
        selected_data['labels'] = labels
        selected_data.to_csv(self.output_data, index=False, encoding='utf_8_sig')

    def calculate_silhouette(self, w2v_vectors, n_clusters):
        """
        计算给定词向量的轮廓系数（Silhouette Coefficient）。
        
        Args:
            w2v_vectors (np.ndarray): 二维numpy数组，表示词向量集合，其中每一行代表一个词向量。
            n_clusters (int): 预期的聚类数量。
        
        Returns:
            float: 轮廓系数的平均值，取值范围为[-1, 1]，值越大表示聚类效果越好。
        
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(w2v_vectors)
        silhouette_avg = silhouette_score(w2v_vectors, cluster_labels)
        return silhouette_avg

    def calculate_wcss(self, w2v_vectors):
        """
        计算WCSS（Within Cluster Sum of Squares）用于评估k-means聚类的效果。
        
        Args:
            w2v_vectors (numpy.ndarray): 形状为(n_samples, n_features)的numpy数组，表示word2vec词向量。
        
        Returns:
            list: 包含每个k值（从2到10）对应的WCSS值的列表。
        
        """
        wcss = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(w2v_vectors)
            wcss.append(kmeans.inertia_)
        return wcss
    
    
    def extract_topics(self):
        """
        从指定的CSV文件中提取每个标签下的主题。
        
        Args:
            无参数。
        
        Returns:
            topics (list): 包含每个标签下主题的列表，每个主题是一个字符串表示，格式为'topic_id topic_terms'。
        
        """
        df = pd.read_csv(self.output_data)
        grouped = df.groupby('labels')
        topics = []

        # 给出HDP的原始输出结果
        for label, group in grouped:
            group = group.dropna(subset=['desc_process'])
            text_list = [str(text).split() for text in group['desc_process']]
            dictionary = Dictionary(text_list)
            corpus = [dictionary.doc2bow(text) for text in text_list]
            # 训练模型
            hdp = HdpModel(corpus, id2word=dictionary)
            topic_result = hdp.print_topics()
            topics.append(topic_result)

            output_file = self.topic_output_file.format(label=label)
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_file, 'w', encoding='utf_8_sig') as f:
                f.write('topic_id, topic_terms\n')
                for topic in topic_result:
                    f.write(f'{topic[0]}, {topic[1]}\n')

# 示例调用
if __name__ == "__main__":
    input_data = 'data/ori_spyder_data.csv'
    threshold = 0.3  # 相似度阈值
    stopwords_file = './hit_stopwords.txt' 
    output_data = './data/selected_data_with_label.csv'
    best_cluster_num = 4
    cluster = Cluster(input_data, threshold,stopwords_file,best_cluster_num, output_data)
    cluster.cluster()
    cluster.extract_topics()
