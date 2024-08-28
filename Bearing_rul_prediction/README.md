# Bearing Remaining-Useful-Life Prediction
This project will keep updating.
- This is a project to predict the remaining useful life of bearings using machine learning techniques.
- The demo file **Bearing_rul_prediction** is an example of generating RUL labels from the health indicator features extracted from the bearing data.
- Finding a suitable strategy to generate precise RUL labels that can represent the degradation trend of the bearing is the key to the success of the RUL prediction model. And it is still studied by scholars and engineers.
- In ***Bearing_rul_prediction***, a precise RUL label generating technique is proposed, which can generate RUL labels that can represent the degradation trend of the bearing accurately.
- A rough explanation of the ***RUL generating technique***: 
  1. Denoise the raw data using wavelet transform
  2. Calculate the statistic features like : mean, std, rms, skewness,kurtosis... from the orginial amplitude data
  2. Smooth the statistic features using savgol_filter
  3. Perform cumulative transformation on the smoothed features, becausing smoothing will reduce some time series information, but the cumulative transformation can regenerate the trend of the time series.
  4. Evaluate the cumulative features based on monotonicity and trend, the higher the score, the more capable that the feature can represent the degenration trend of the bearing.
  5. Select the features with high scores, perform feature fusion, so that the fused feature will have the most accurate and generalized representation of the bearing degradation trend.
  6. Min-max transform the degredation trend feature to the range of [1,0], the closer to 1, the more severe the bearing is. ***And this will be the label of the RUL.***
- The ***Bearing_rul_prediction*** is a demo file to Predict the bearing RUL in XJTU-SY dataset, Other datasets like ***IMS***,***CoE** and ***real lab data*** are already tested by the author if the data is preprocessed correctly and can be applied to this file with amendments.
- Models used include: LinearRegression,RandomForest, Support Vector Regression,KNN Regression, LSTM, CNN,ResNet, Transformer...
- **!!! Please note that the time series data in one sample file should be stored in one row, and vertically stacked in one csv/mat/xlsx... file.**
### For code explanation, datasets and ideas sharing, please contact the author at ***[ianwu0907@gmail.com](mailto:)***.