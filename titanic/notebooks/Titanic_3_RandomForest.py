import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# 挑選特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

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

train_df['Embarked'] = train_df['Embarked'].map({'C':0, 'Q':1, 'S':2})
test_df['Embarked'] = test_df['Embarked'].map({'C':0, 'Q':1, 'S':2})



X = train_df[features]
y = train_df['Survived']

X_test = test_df[features]



# 分訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = RandomForestClassifier(
    n_estimators=100,   # 樹的數量
    max_depth=5,       # 樹的最大深度 (避免過擬合)
    random_state=42
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