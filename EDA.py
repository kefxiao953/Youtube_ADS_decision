# 引入模块
import pandas as pd
import arrow
from matplotlib import pyplot as plt

#==========================================EDA============================================
#EDA是我们进行数据项目的第一步，在进行分析之前，首先我们应该去了解数据，这就是EDA（探索性数据分析），
#需要了解的，一般包括以下这些内容：
#1.基础信息：数据量（有多少行）,数据特征(有多少列，每一列表示什么意思，每一列的数据类型),
#           数据意义(每行数据表示的含义）
#2.数据分布：了解一下Numerical数据列的数据分布，包括中位数，2分位数，4分位数，方差，标准差等等
#3.统计信息：对影响业务的数据列进行统计。
#3.数据问题：哪些数据列存在异常，什么样的异常。
#4.数据间的关系：行与行之间的关系，列与列之间的关系
#5.根据本项目的业务目标，探索数据其他信息
#=========================================================================================
#读取数据
data = pd.DataFrame()
df = pd.read_csv(r'USvideos.csv')
data = pd.concat([df, data], ignore_index=True)

#=======================观察基础信息=============================
data.info() 

#=======================观察数据分布=============================

views_trend = {}
for i in range(1,101,1):
    views_trend[i] = data['views'].quantile(i/100)

#=======================计算统计信息=============================
#这里首先我想看的是按月合计的上榜视频数量
#整理数据格式
data['trending_date1'] = data['trending_date'].apply(lambda x: arrow.get(str('20'+x[:2]+'-'+x[6:8]+'-'+x[3:5])))
data['trending_year'] = data['trending_date1'].apply(lambda x: x.year)
data['trending_month'] = data['trending_date1'].apply(lambda x: x.month)

#计算每月上trending榜的视频数量
video_num = data.groupby(['trending_month'])['video_id'].count().reset_index()

#计算每月trending视频和总trending视频的数量比
video_num['rate'] = video_num['video_id']/data['video_id'].shape[0]

#对每个category上的浏览量views进行求和，找出各个category的浏览量总数
views_num_category = pd.pivot_table(data,index='category_id', values='views',aggfunc='sum').reset_index().sort_values(by='views')

#被最多喜欢的视频是哪个及视频链接缩略图
most_like = data.groupby(['category_id','thumbnail_link'])['likes'].sum().reset_index().sort_values(by='likes',ascending=False)

#流行视频中标签最多的词是什么
tags_num = data['tags'].value_counts()

#=======================观察数据问题=============================
data.describe()

#=======================探索数据列的关系=============================
#观看数、喜欢、不喜欢、评论数的相关性？
releation = data[['views', 'likes', 'dislikes', 'comment_count']].corr()
releation = data[['views', 'trending_month']].corr()
import seaborn as sns
ax = sns.heatmap(releation)
plt.show()

#=======================根据业务，进行相应的数据探索============================

#分析 “在什么时间点上投流”这个问题，需要根据数据，理解项目的意图
#一开始，可能会尝试使用时间序列去预测views值，但会发现两者相关性极低，无法预测。
#于是需要重新考虑“在什么时间点上投流”，根据月份，年？都不合适
#最终找到，是从publish time到trending_date之间的间隔时间内，选择合适的时间点进行投流，且超出7天的视频，上榜率就较低，因此就不值得投流了。
#这样就获得了数据层面的洞察
data["trending_date"] = pd.to_datetime(data["trending_date"],format ="%y.%d.%m")
data["publish_time"]=pd.to_datetime(data["publish_time"],format ="%Y-%m-%dT%H:%M:%S.%fZ")

data["timedelta"]=(data["trending_date"]- data["publish_time"]).dt.days
data["time_cut"]=pd.cut(data["timedelta"],[0,1,3,5,7,3650],labels=["0-1 days","1-3 days","3-5 days","5-7 days","7 +"])
time_pie=data.groupby("time_cut").timedelta.count()
time_pie.plot(kind = "pie", autopct='%.2f%%', fontsize=20, figsize=(16, 9))
plt.title("上榜时间")
plt.show()



