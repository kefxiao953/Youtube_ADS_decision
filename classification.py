
# 引入模块
import pandas as pd
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

#================================================================
#进行word2vec转词向量的部分与clustering部分一样
#================================================================
# 文本预处理函数
def preprocess_text(text):
    text = text.lower()  # 转换为小写
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    words = text.split()  # 分词
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # 去除停用词
    return words

# 读取数据
df = pd.read_csv('USvideos.csv')


# 预处理文本数据
processed_docs = df['title'].apply(preprocess_text)
processed_docs = df['channel_title'].apply(preprocess_text)
processed_docs = df['tags'].apply(preprocess_text)
#processed_docs = df['description'].apply(preprocess_text)

# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences=processed_docs, vector_size=100, window=5, min_count=1, workers=4)

# 获取每个文档的词向量
def get_document_vector(doc):
    vector = [word2vec_model.wv[word] for word in doc if word in word2vec_model.wv]
    return np.mean(vector, axis=0) if len(vector) > 0 else np.zeros(100)

doc_vectors = processed_docs.apply(get_document_vector).tolist()
print(doc_vectors)

# 转换为标准化格式
scaler = StandardScaler()
X = scaler.fit_transform(doc_vectors)
print(X)

#================================================================
#现在我们开始进行classification
#首先我们尝试一下耳熟能详的LogisticRegression
#================================================================
# 引入必要的模块
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 将处理后的特征与目标标签进行整合
df['doc_vector'] = doc_vectors
df.dropna(subset=['category_id', 'doc_vector'], inplace=True)
X = np.array(df['doc_vector'].tolist())
y = df['category_id']

# 拆分数据集进行训练和测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 选择分类模型
model = LogisticRegression(max_iter=1000)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型性能
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

#================================================================
#这是因为Logistic regression对多分类问题的适用性不足
#让我们把问题转化为二分类再试一下。
#请在df['doc_vector'] = doc_vectors代码后方加入这句代码：
# df = df[df['category'].isin([2, 10])]
#然后其他代码不变，再跑一次
#有没有发现accuracy_score提高了很多？这是因为LogisticRegression的二分类特性所决定的。
#================================================================
#回到多分类问题，我们尝试使用其他分类模型，比如Random forest和Xgboost看看效果
#================================================================

from sklearn.ensemble import RandomForestClassifier
# 选择分类模型
RandomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
RandomForestModel.fit(X_train, y_train)

# 进行预测
y_pred = RandomForestModel.predict(X_test)

# 评估模型性能
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))


