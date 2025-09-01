import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 載入資料
train_df = pd.read_csv('titanic/data/train.csv')
test_df = pd.read_csv('titanic/data/test.csv')

# 查看前幾列
#print(train_df.head())

# 檢查缺失值
#print(train_df.isnull().sum())
#原本age 177, cabin 687, embarked 2
#print(test_df.isnull().sum())
#原本age 86, cabin 327, fare 1



# 填補 Age 缺失值
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age']  = test_df['Age'].fillna(test_df['Age'].median())

# 填補 Embarked 缺失值
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

# 填補 Fare 缺失值 (test.csv 有一筆 NaN)
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# 類別編碼
train_df['Sex'] = train_df['Sex'].map({'male':0, 'female':1})
test_df['Sex'] = test_df['Sex'].map({'male':0, 'female':1})

# Embarked one-hot encoding
train_df = pd.get_dummies(train_df, columns=["Embarked"])
test_df = pd.get_dummies(test_df, columns=["Embarked"])

# 對齊欄位（補上缺失的 dummy columns）
train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

# 挑選特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
             'Embarked_C', 'Embarked_Q', 'Embarked_S']

X = train_df[features]
y = train_df['Survived']
X_test = test_df[features]



# 分訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = XGBClassifier(
    n_estimators=500,     # 樹的數量
    max_depth=4,          # 樹的深度
    learning_rate=0.01,   # 學習率 (小一點避免過擬合)
    subsample=0.8,        # 每次隨機抽樣比例
    colsample_bytree=0.8, # 特徵隨機抽樣比例
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

#產生提交檔案
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': model.predict(X_test)
})

submission.to_csv('titanic_submission.csv', index=False)
print("Submission file created!")


# 顯示重要特徵
'''
import matplotlib.pyplot as plt
import seaborn as sns

feature_importances = pd.Series(model.feature_importances_, index=features)
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance")
plt.show()
'''