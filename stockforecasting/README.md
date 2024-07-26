# Stock-forecasting

This project is a stock forecasting project that uses LSTM to predict the future stock prices.
-  ***Stockcrawler.py*** is a python file that uses the re,requests and pandas to get the selected stock data.
-  Run any of the ***"Stockforecasting"*** files to get the stock forecasting results
-  input the stock code and the length of time sequences for LSTM(suggest to use 30 = one month) 
-  to predict the future stock prices.
-  Stockcrawler will collect the stock data from eastmoney.com and save it as a excel file.
-  The stock data has the following feature columns: 'date','open','close','high','low','vol','amount(thousand)','amplitude(%)',pct_chg(%)','change'
-  Then the program will output the total number of stock data collected from the stockcrawler.py
-  The User will be asked to select the percentage formation of the training data and the testing data. (suggested to use 0.7)
-  Then the Program will start to pre-train the rnn models with the stock datas collected
-  After the pre-training is done, the user can input the number of days that he/she wants to predict.
-  The program will output the predicted 'open','high','low','close' values of the predicted stock.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ***"stockforecasting_pre_trainning+transfer_learning.py"*** is a file that pre-trains an RNN model with 2 stocks and use the model directly to predict the 3rd stock that the user selected.
### ***"stockforecasting_pre_trainning_v2.py"*** pre-trains one RNN model with 3 stocks,including the stock that the user selected to predict.
### ***"stockforecasting_weighted_average.py"*** will pre-train 2 RNN models with 2 stocks and use the weighted average of the 2 models to predict the 3rd stock that the user selected.
### ***"stockforecasting_weighted_average+pre_training"*** will pre-train 2 RNN models with 3 stocks,including the stock that the user selected to predict and use the weighted average of the 2 models to predict the 3rd stock that the user selected.

---

- Based on testing results,***"stockforecasting_pre_trainning_v2.py"*** has the best performance, with the lowest relative train/test errors and best generalization ability.
- ***"stockforecasting_weighted_average+pre_training.py"*** is considered to have the greatest potential, if the running time and calculation power is enough. Because it could possibly be further generalized to pre-train multiple RNN models and use the weight average to increase the generalization ability and accuracy.
- The generalized version of ***"stockforecasting_weighted_average+pre_training.py"*** is ***"stockforecasting_weighted_average+pre_training_v2.py"***, is still under development.

The repository already included some collected stock data for testing.

---
To run the program, you need to install the following packages:
-  numpy
- pandas
- matplotlib
- tensorflow
- Pytorch
- scikit-learn
- requests
- re
- datetime

***

# ***UPDATES***:

- 2024/7/26 uploaded this repository to ***myproject***