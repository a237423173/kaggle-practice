import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# =====================
# 缺失值處理
# =====================
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age']  = test_df['Age'].fillna(test_df['Age'].median())

train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# =====================
# 類別編碼 (Sex + Embarked)
# =====================
train_df['Sex'] = train_df['Sex'].map({'male':0, 'female':1})
test_df['Sex'] = test_df['Sex'].map({'male':0, 'female':1})

train_df = pd.get_dummies(train_df, columns=["Embarked"])
test_df = pd.get_dummies(test_df, columns=["Embarked"])

# =====================
# 特徵工程
# =====================

# 1. Title (從姓名中擷取稱謂)
for df in [train_df, test_df]:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
    )
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Title'] = df['Title'].map({'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Rare':4}).fillna(4)

# 2. FamilySize
for df in [train_df, test_df]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 3. IsAlone
for df in [train_df, test_df]:
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 4. Cabin Indicator
for df in [train_df, test_df]:
    df['CabinIndicator'] = df['Cabin'].notnull().astype(int)

# 5. Age*Class
for df in [train_df, test_df]:
    df['Age*Class'] = df['Age'] * df['Pclass']

# =====================
# 對齊欄位
# =====================
train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

# =====================
# 選擇特徵
# =====================
features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked_C', 'Embarked_Q', 'Embarked_S',
    'Title', 'FamilySize', 'IsAlone', 'CabinIndicator', 'Age*Class'
]


X = train_df[features]
y = train_df['Survived']
X_test = test_df[features]



# 分訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = XGBClassifier(
    n_estimators=400,     # 樹的數量
    max_depth=4,          # 樹的深度
    learning_rate=0.01,   # 學習率 (小一點避免過擬合)
    subsample=0.8,        # 每次隨機抽樣比例
    colsample_bytree=0.8, # 特徵隨機抽樣比例
    random_state=42,
    eval_metric="logloss"
)


# 指定驗證集，讓模型在每輪後計算驗證集 loss / accuracy
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
results = model.evals_result()

# 畫圖
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

# Logloss 曲線
plt.figure(figsize=(10,4))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
plt.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()


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

