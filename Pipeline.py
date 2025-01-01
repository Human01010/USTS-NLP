import jieba
import jieba.posseg as pseg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud
import requests
from lxml import etree
from snownlp import SnowNLP  # 导入SnowNLP库
from transformers import pipelines
from collections import Counter

# 请求弹幕XML数据
url = 'https://comment.bilibili.com/26247693358.xml'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.encoding = 'utf-8'
    response.raise_for_status()  # 如果请求不成功，会抛出异常
    xml_content = response.text  # 如果请求成功，获取返回的XML内容
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
    xml_content = None  # 失败时设为None

if xml_content:  # 仅在请求成功时继续处理
    # 解析弹幕XML
    root = etree.fromstring(xml_content.encode('utf-8'))
    danmus = root.xpath('//d/text()')

    # 使用SnowNLP进行情感分析的函数，并根据词性赋予不同权重
    def sentiment_analysis(text):
        s = pipelines(text)
        return s.sentiments  # 返回情感得分，范围是0到1，值越大表示情感越正面

    # 计算每条弹幕的情感得分
    def calculate_sentiments(danmus):
        sentiments = []
        for danmu in danmus:
            s = SnowNLP(danmu)
            sentiments.append(s.sentiments)  # 记录情感得分
        return sentiments

    # 获取情感得分
    danmaku_sentiments = calculate_sentiments(danmus)

    # 绘制情感得分分布
    plt.figure(figsize=(8, 6))
    sns.histplot(danmaku_sentiments, bins=50, kde=True, color="blue")
    plt.title("Sentiment Score Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

    # 绘制正负面情感比例
    def plot_sentiment_pie(sentiments):
        positive = sum(1 for s in sentiments if s > 0.5)
        negative = len(sentiments) - positive

        labels = ['Positive', 'Negative']
        sizes = [positive, negative]
        colors = ['#66b3ff', '#ff9999']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title("Positive vs Negative Sentiments")
        plt.axis('equal')
        plt.show()

    plot_sentiment_pie(danmaku_sentiments)

    # 添加词云以显示主要关键词
    positive_text = ' '.join([danmu for danmu, sentiment in zip(danmus, danmaku_sentiments) if sentiment > 0.5])
    negative_text = ' '.join([danmu for danmu, sentiment in zip(danmus, danmaku_sentiments) if sentiment <= 0.5])

    positive_wordcloud = WordCloud(font_path='C:/Windows/Fonts/msyh.ttc', width=800, height=400, background_color='white').generate(positive_text)
    negative_wordcloud = WordCloud(font_path='C:/Windows/Fonts/msyh.ttc', width=800, height=400, background_color='white').generate(negative_text)

    # 绘制正面情感词云
    plt.figure(figsize=(10, 6))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Wordcloud of Positive Sentiments')
    plt.show()

    # 绘制负面情感词云
    plt.figure(figsize=(10, 6))
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Wordcloud of Negative Sentiments')
    plt.show()

else:
    print("没有获取到有效的弹幕数据。")
