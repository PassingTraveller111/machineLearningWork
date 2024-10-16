import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
# 本节实验尝试使用Scikit-learn库来实现逻辑回归
# Scikit-learn 是机器学习领域中非常流行和实用的工具库，广泛应用于各种数据分析和建模任务中。

from sklearn.linear_model import LogisticRegression

#下面的代码从scikit-learn导入逻辑回归模型。您可以通过调用fit函数将此模型拟合到训练数据上。
lr_model = LogisticRegression()
lr_model.fit(X, y)

# 通过predict函数作出预测
y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)

# 通过调用score函数来计算此模型的准确性。
print("Accuracy on training set:", lr_model.score(X, y))
