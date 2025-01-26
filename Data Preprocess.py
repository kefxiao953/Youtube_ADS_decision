#======================Data Preprocess==============================
#根据数据中存在的问题，进行相应数据预处理，将数据整理为干净，无错误的数据
#通常，我们会对数据列中的异常值，缺失值，空值做针对性处理
#在本项目中，EDA阶段观察到的数据质量较好（只有数据的desciption列有一部分数据缺失，但这不影响我们的主要业务，可以忽略），
#因此，在数据完整性方面不需要做过多的数据预处理。
#我们可以focus on的数据的一致性，具体来讲，就是找出数据中存在的Outlier，进行Outlier Removal

#导入常用库
import pandas as pd
import numpy as np
import arrow
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor

#=======================Outlier Removal==========================、
#要进行outlier的去除，首先要发现outlier的存在，这一点需要在EDA阶段做到。
#EDA阶段，我们发现Views和likes，dislike，comments之间存在一定的线性关系。
#而观察数据+常识，让我们知道，同一个category之下，views和likes的比例应该是大致相同的
#因此我们可以通过对views和like之间的比例进行建模，来拟合大多数数据间，
#再通过该模型的预测来把预测值和真实值差距较远的数据给标记为Outlier，并剔除掉。

# 将数据集拆分为训练集和测试集，按照8:2拆分。
train_data = data[:40000]
test_data = data[40000:]

# 提取特征和目标变量
#X_train = data[["likes","dislikes","comment_count"]].to_numpy()
X_train = train_data[["likes"]].to_numpy()
Y_train = train_data["views"].to_numpy()
X_test = test_data[["likes"]].to_numpy()
Y_test = test_data["views"].to_numpy()

# 定义model为Linear Regression模型，model2为Decision Tree模型并进行训练和预测
model = LinearRegression()
model2 = DecisionTreeRegressor(random_state=42)
model.fit(X_train, Y_train)
model2.fit(X_train, Y_train)

# 输出Linear Regression模型的评估指标，如均方误差和R2得分等
Y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("MSE: %.2f" % mse)
print("R2 Score: %.2f" % r2)

# 输出Decision Tree模型的评估指标，如均方误差和R2得分等
Y_pred2 = model2.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse2 = mean_squared_error(Y_test, Y_pred2)
r22 = r2_score(Y_test, Y_pred2)
print("MSE2: %.2f" % mse2)
print("R22 Score: %.2f" % r22)

#通过阈值法去除离群值
Y_train_pred = model.predict(X_train)
residuals = Y_train - Y_train_pred
studentized_residuals = residuals / np.sqrt(mean_squared_error(Y_train, Y_train_pred))
outliers = np.abs(studentized_residuals) > 2
plt.scatter(X_train, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('X_train')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

#计算cook distance
cook_distance = (residuals ** 2) / (model.coef_ ** 2 * X_train.var())
outliers_cooks = cook_distance > 4 / len(X_train)

# 打印出离群值的索引
print("Outliers based on studentized residuals:", np.where(outliers)[0])
print("Outliers based on Cook's distance:", np.where(outliers_cooks)[0])

outliers_indices = np.where(outliers)[0]

#删除离群数据
cleaned_X = np.delete(X_train, outliers_indices, axis=0)
cleaned_y = np.delete(Y_train, outliers_indices, axis=0)

# 重新拟合线性回归模型
model.fit(cleaned_X, cleaned_y)
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("new MSE: %.2f" % mse)
print("new R2 Score: %.2f" % r2)
#可以发现，重新拟合后，mse有一定的下降。


