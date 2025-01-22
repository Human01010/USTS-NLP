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

# 假设我们已经加载了弹幕数据
url = 'https://comment.bilibili.com/26247693358.xml'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 请求弹幕XML数据
try:
    response = requests.get(url, headers=headers, timeout=10)
    response.encoding = 'utf-8'
    response.raise_for_status()
    xml_content = response.text
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")

# 解析弹幕XML
root = etree.fromstring(xml_content.encode('utf-8'))
danmus = root.xpath('//d/text()')

# 使用SnowNLP进行情感分析的函数，并根据词性赋予不同权重
def sentiment_analysis(text):
    s = SnowNLP(text)
    return s.sentiments  # 返回情感得分，范围是0到1，值越大表示情感越正面

# 计算词性权重
def calculate_pos_weight(pos):
    # 常见词性及其对应的权重
    pos_weights = {
        'a': 1.0,  # 形容词
        'n': 0.8,  # 名词
        'v': 1.2,  # 动词
        'd': 0.9,  # 副词
        'p': 1.1,  # 介词
        'c': 0.7,  # 连词
        'r': 1.3,  # 代词
        'm': 1.1   # 数词
    }
    return pos_weights.get(pos, 1.0)  # 默认权重为1.0

# 示例情感标签（0为负面，1为正面），假设标记所有弹幕为正面或负面
def get_weighted_labels(danmus):
    labels = []
    for danmu in danmus:
        words = pseg.cut(danmu)
        weighted_sentiment = 0
        for word, pos in words:
            sentiment_score = sentiment_analysis(word)
            pos_weight = calculate_pos_weight(pos)
            weighted_sentiment += sentiment_score * pos_weight
        labels.append(1 if weighted_sentiment > 0.5 else 0)
    return labels

labels = get_weighted_labels(danmus)

# 数据预处理：分词并去除停用词
def load_stopwords(filepath='C:/pyprojects/lessons/NLP/hit_stopwords.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    return stopwords

def preprocess_text(text, stopwords):
    words = pseg.cut(text)
    filtered_words = [word for word, pos in words if word not in stopwords and len(word) > 1]  # 去掉停用词和单字
    return ' '.join(filtered_words)

# 加载停用词表
stopwords = load_stopwords('C:/pyprojects/lessons/NLP/hit_stopwords.txt')

# 对弹幕进行预处理
danmus_processed = [preprocess_text(danmu, stopwords) for danmu in danmus]

# 使用TF-IDF进行特征提取
vectorizer = TfidfVectorizer(max_features=5000)  # 设置最大特征数为5000
X = vectorizer.fit_transform(danmus_processed).toarray()
y = np.array(labels)

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练支持向量机（SVM）模型
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出分类报告
print("SVM模型评估：")
print(classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM模型准确率: {accuracy * 100:.2f}%")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.title('confusion matrix of SVM model')
plt.xlabel('prediction labels')
plt.ylabel('true labels')
plt.show()

# 绘制词云
all_text = ' '.join(danmus_processed)
wordcloud = WordCloud(font_path='C:/Windows/Fonts/msyh.ttc', width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('wordcloud of keywords')
plt.show()

############################################################################################################################
#logistic regression
from sklearn.linear_model import LogisticRegression

# 使用逻辑回归替换SVM模型
model = LogisticRegression(max_iter=1000)  # 设置最大迭代次数，防止模型收敛不完全
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出分类报告
print("LogisticRegression模型评估：")
print(classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"LogisticRegression模型准确率: {accuracy * 100:.2f}%")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.title('confusion matrix of LogisticRegression model')
plt.xlabel('prediction labels')
plt.ylabel('true labels')
plt.show()

############################################################################################################################
#Random Forest
from sklearn.ensemble import RandomForestClassifier

# 使用随机森林替换SVM模型
model = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators为决策树的数量
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出分类报告
print("RF模型评估：")
print(classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"RF模型准确率: {accuracy * 100:.2f}%")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.title('confusion matrix of Random Forest model')
plt.xlabel('prediction labels')
plt.ylabel('true labels')
plt.show()

############################################################################################################################
#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

# 使用梯度提升树替换SVM模型
model = GradientBoostingClassifier(n_estimators=100, random_state=42)  # n_estimators为弱分类器的数量
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出分类报告
print("GB模型评估：")
print(classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"GB模型准确率: {accuracy * 100:.2f}%")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.title('confusion matrix of Gradient Boosting model')
plt.xlabel('prediction labels')
plt.ylabel('true labels')
plt.show()

############################################################################################################################
#XGBoost
import xgboost as xgb

# 使用XGBoost替换SVM模型
model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出分类报告
print("XGBoost模型评估：")
print(classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost模型准确率: {accuracy * 100:.2f}%")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.title('confusion matrix of XGBoost model')
plt.xlabel('prediction labels')
plt.ylabel('true labels')
plt.show()
