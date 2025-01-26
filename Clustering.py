#======================Clustering==============================
#目前为止，我们为了确定对哪一类视频进行投流，已经做了很多的工作
#但是，我们有没有想过，所有的工作，都是基于已有的category。
#我们不禁要问，是否也存在不属于特定category的的视频，值得我们投流呢？从常识上说，这是肯定的。
#那么从数据层面上，我们应该怎么把这一类视频给找出来呢？
#这就要用到我们经常听说，但很少用到的数据模型，clutering。它可以帮我们无监督地找出各种存在内部联系的数据，
#并将它们聚为多个cluster
#有很多中cluster方法，我们这一次使用k-means方法。
#====================================================
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
print("============================================")
print(doc_vectors)

# 转换为标准化格式
scaler = StandardScaler()
X = scaler.fit_transform(doc_vectors)
print("**********************************************")
print(X)
#==========================确定k值============================
#词向量构建好了，下面我们要确定k-means算法的k值，一般用肘部法。
# 计算不同 k 值下的 SSE(Sum of Squared Errors, SSE）
sse = []
k_values = range(1, 11)  # 你可以根据需要调整 k 值的范围
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# 绘制肘部法图形
plt.figure(figsize=(10, 7))
plt.plot(k_values, sse, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method For Optimal k')
plt.show()

#==========================思考题==============================
#如果没有明显的elbow，那么如何确定k值？

# 使用K-means算法进行聚类
num_clusters = 5  # 我们此处设K值为5
# 训练一个k-means模型
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# 拟合
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.labels_

# 使用PCA降维以便可视化（可选，目的在于将原始cluster映射到2维平面上，方便画图）
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# 可视化聚类结果
plt.figure(figsize=(10, 7))
colors = ['r', 'g', 'b', 'c', 'm']
for i in range(num_clusters):
    points = principal_components[labels == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Cluster {i+1}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters of Documents')
plt.legend()
plt.show()

#===========================预测新的数据属于哪个cluster===========================
# 新数据，这里只是一个范例，你应该改成我们的.csv文件中的数据格式
new_text = "Your new text data here"
# 预处理新数据
new_processed = preprocess_text(new_text)

# 获取新数据的词向量
new_vector = get_document_vector(new_processed)

# 标准化新数据
new_vector_scaled = scaler.transform([new_vector])

# 预测新数据的聚类
new_label = kmeans.predict(new_vector_scaled)
print(f'The new data point belongs to cluster: {new_label[0]}')
