## Kaggle 練習紀錄

- [Titanic: Machine Learning from Disaster](./titanic)  
  - 模型：Logistic Regression
  - Public LB: 0.76315  
  - 模型：Logistic Regression + title提取
  - Public LB: 0.77272
  - 模型：RandomForest
  - Public LB: 0.78229
  - 模型：RandomForest + familysize
  - Public LB: 0.78229 (這樣代表RF可能已經有算到familysize權重)
  - 模型：RandomForest + title
  - Public LB: 0.79186
  - 模型：XGBoost
  - Public LB: 0.77272
  - 模型：XGBoost + title + familysize + cabin + age*class
  - Public LB: 0.76076 (overfitting，仍需調參數)