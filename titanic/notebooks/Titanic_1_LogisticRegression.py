import pandas as pd

# 載入資料
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 查看前幾列
#print(train_df.head())

# 檢查缺失值
#print(train_df.isnull().sum())
#原本age 177, cabin 687, embarked 2
#print(test_df.isnull().sum())
#原本age 86, cabin 327, fare 1


# 簡單統計描述
#print(train_df.describe())

# 觀察性別與生存率關係
'''
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.show()
'''

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


# 特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']

X_test = test_df[features]




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 分訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = LogisticRegression(max_iter=1000)
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

