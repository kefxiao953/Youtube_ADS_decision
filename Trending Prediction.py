#======================Trending prediction==============================
#我们为了选择某个category进行投流，需要考虑多个因素。
#目前我们已知的有：category的基础流量数据，category与业务的契合程度
#我们现在要考虑第三个纬度，未来category的流量预测值的大小
#我们采用Time Series Prediction来完成这个任务
#常见的Time Series Prediction model 有很多，答案中采用ARIMA。
#同学可以自行尝试LSTM
#====================================================
# 引入模块
import pandas as pd
import numpy as np
import arrow
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 读取数据，假设数据包含两个特征 X1 和 X2，以及目标变量 Y
data = pd.DataFrame()
df = pd.read_csv(r'USvideos.csv')
data = pd.concat([df, data], ignore_index=True)
data['trending_date'] = data['trending_date'].apply(lambda x: arrow.get(str('20'+x[:2]+'-'+x[6:8]+'-'+x[3:5])))
data['trending_year'] = data['trending_date'].apply(lambda x: x.year)
data['trending_month'] = data['trending_date'].apply(lambda x: x.month)
reference_date = arrow.get('2017-01-01')
data['days_since_start'] = data['trending_date'].apply(lambda x: pd.Timedelta(days=(x - reference_date).days))
data['days_since_start'] = data['days_since_start'].dt.days

#我们先对cotegory_id为2的类别进行考量
music_data = data[(data['category_id'] == 2)].reset_index() #这个地方要注意，必要写成'2'，否则表示的是字符串'2'
print(music_data)

# 将数据整理为时间序列
views_num_days = pd.pivot_table(music_data,index='days_since_start', values='views',aggfunc='sum').reset_index().sort_values(by='views')
views_num_days = views_num_days.sort_values(by='days_since_start')

# 将数据集拆分为训练集和测试集
# 总数为134条，因此按8:2，切分点为107
train_data = views_num_days[:107]
test_data = views_num_days[107:]

# 整理train/test中的自变量和因变量
X_train = train_data["days_since_start"].to_numpy().reshape(-1,1)
Y_train = train_data["views"].to_numpy()
X_test = test_data["days_since_start"].to_numpy().reshape(-1,1)
Y_test = test_data["views"].to_numpy()

# 尝试使用线性回归模型，进行训练和预测并观察效果
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Y_pred: " + str(Y_pred))
print("Y_test: " + str(Y_test))
# 输出模型的评估指标，如均方误差和R2得分等
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("MSE: %.2f" % mse)
print("R2 Score: %.2f" % r2)

# 换用ARIMA进行操作
# 此处为重点！需要知道ARIMA的三个参数pdq分别表示什么含义。
p = 2  # AR 阶数
d = 1  # 差分阶数
q = 2  # MA 阶数

# 创建并拟合ARIMA模型
model = ARIMA(Y_train, order=(p, d, q))
model_fit = model.fit()

# 预测
predictions = model_fit.forecast(steps=len(X_test))

# 可视化预测结果
plt.plot(Y_train, label='Train')
plt.plot(range(len(Y_train), len(Y_train) + len(Y_test)), Y_test, label='Test')
plt.plot(range(len(Y_train), len(Y_train) + len(Y_test)), predictions, label='Predictions')
plt.xlabel('Days')
plt.ylabel('Views')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()
#反复调pdq的值，找到最佳拟合曲线
