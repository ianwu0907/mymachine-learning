# mymachine-learning————这是一个记录我的计算机学习过程的git文件夹
- 2024/7/22 更新 -- 通过bs4,requests和re库实现的豆瓣电影网站爬虫 -- doubancrawl.py
- 2024/7/23 更新 -- 通过re,requests 和pandas实现东方财富网股票数据的爬虫 -- 东方财富网股票爬虫.py
- 2024/7/27 更新文件夹 ***stockforecasting*** -- 一个使用LSTM预测股票价格的项目
- 2024/7/27 对***stockforecasting***文件夹下的文件进行了更新
- 2024/7/31 更新 -- ***stockforecasting***，添加了prodit-predic算法。
- 2024/7/31 更新 -- ***Bearing-diagnostic***，添加了模型以使用CWRU数据集分类10种故障类型。
- 2024/8/1 更新 -- ***Bearing-diagnostic***，添加了数据预处理和特征提取文件***data_create_CWRU_FFT.ipynb***，该文件对CWRU数据集进行快速傅里叶变换，然后进行包络谱分析，提取3个特征频率数据。
- 2024/8/1 更新 -- ***Bearing-diagnostic***，添加了文件***model_train_FFT.ipynb***，该文件在从CWRU数据集中提取的特征频率数据上训练机器学习模型。
- 2024/8/28 更新 -- ***Bearing-diagnostic***，添加了新的训练文件，将振幅数据首先转换为2D，在***IMS和CWRU数据集***上进行轴承故障诊断测试，整体表现获得了提升。
- 2024/8/28 创建***Bearing_rul_prediction***文件夹，添加了demo文件，用于预测轴承的剩余寿命,并使用了一种准确且泛化性高的RUL标签生成方式。
---
# Author's Note
- Most of the files in this repository are written in Chinese. But if there is a need, I could translate them into English. (Anyway I would probably update the English version on my self in the future)
- For code explanation, datasets and ideas sharing, please contact the author at ***[ianwu0907@gmail.com](mailto:)***.
---
# mymachine-learning — This is a Git folder that documents my computer learning journey
- 2024/7/22 Update -- A web crawler for ***Douban movie website*** implemented using bs4, requests, and re libraries -- doubancrawl.py
- 2024/7/23 Update -- A web crawler for ***stock data*** from Eastmoney.net implemented using re, requests, and pandas -- Eastmoney Stock Crawler.py
- 2024/7/27 Update to the repository ***stockforecasting*** -- A project that predicts stock prices using LSTM
- 2024/7/27 Files under the ***stockforecasting*** repository have been updated
- 2024/7/31 Update ***stockforecasting*** , added prodit-predic algorithm
- 2024/7/31 Update ***Bearing_diagnostic*** , added maching-models to classify 10 types of faults with CWRU dataset.
- 2024/8/1 Update ***Bearing_diagnostic*** , added data pre-processing and feature extraction file ***data_create_CWRU_FFT.ipynb***, which performs fast-fourier transformation and then Envelope Spectrum Analysis on the CWRU dataset, extracting 3 feature frequency data.
- 2024/8/1 Update ***Bearing_diagnostic*** , added file ***model_train_FFT.ipynb***, which trains the machine-learning models on the feature frequency data extracted from the CWRU dataset.
- 2024/8/28 Update ***Bearing_diagnostic***, added new training file that transforms the amplitude data into 2D first and give a better overall performance on ***IMS and CWRU dataset*** bearing fault diagnostic tasks.
- 2024/8/28 Create ***Bearing_rul_prediction*** repository, added demo file to predict the remaining useful life of bearings on different datasets with a special and robust RUL label generating technique.