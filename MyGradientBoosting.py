import pandas as pd
import numpy as np
import random
from MyTreeReg import MyTreeReg


class MyBoostReg():

    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, loss='MSE', metric=None,
                 max_features=0.5, max_samples=0.5, random_state=42, reg=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.loss = loss
        self.metric = metric
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.pred_0 = None
        self.trees = []
        self.sum_preds = 0
        self.best_score = None
        self.udpate_loss()
        self.update_metric()
        self.dynamic_learning_rate()
        self.count_all_leafes = 0
        self.fi = {}

    def __str__(self):
        return f'MyBoostReg class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
    
    def fit(self, X, y, verbose=None):
        self.cols = X.columns.values.tolist()
        self.fi.update({col: 0.0 for col in self.cols})
        self.cols_count = len(self.cols)
        self.N = X.shape[0]
        random.seed(self.random_state)
        self.pred_0 = self.first_pred(y)
        self.sum_preds += np.full(self.N, self.pred_0)
        for i in range(1, self.n_estimators + 1):
            cols_learn = random.sample(self.cols, round(self.cols_count * self.max_features))
            rows_idx = random.sample(range(self.N), round(self.N * self.max_samples))
            X_learn = X[cols_learn]
            X_learn = X_learn.iloc[rows_idx]
            # y_learn = y.iloc[rows_idx]
            error_target = -self.grad(self.sum_preds, y)[rows_idx]
            model = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins)
            model.fit(X_learn, error_target, N=self.N)
            self.change_leafs_values(y, model.leafs)
            self.count_all_leafes += model.leafs_cnt
            self.sum_preds += self.learning_rate(i) * model.predict(X)
            self.trees.append(model)
            error = self.compute_loss(y, self.sum_preds)
            if verbose:
                if i % verbose == 0:
                    print(self.compute_loss(y, self.sum_preds))
        if self.metric:
            self.best_score = self.metric(y, self.sum_preds)
        else:
            self.best_score = error
        self.result_feature_importance()

    def predict(self, X):
        res = self.pred_0
        for i, tree in enumerate(self.trees):
            res += self.learning_rate(i + 1) * tree.predict(X)
        return res
    
    def mse_grad(self, y_pred, y):
        return 2 * (y_pred - y)
    
    def mae_grad(self, y_pred, y):
        return np.sign(y_pred - y)
    
    def udpate_loss(self):
        if self.loss == 'MSE':
            self.compute_loss = self.mse
            self.grad = self.mse_grad
        elif self.loss == 'MAE':
            self.compute_loss = self.mae
            self.grad = self.mae_grad
        else:
            raise Exception('Нет такого лосса')
        
    def update_metric(self):
        if self.metric == 'MAE':
            self.metric = self.mae
        elif self.metric == 'MSE':
            self.metric = self.mse
        elif self.metric == 'RMSE':
            self.metric = self.rmse
        elif self.metric == 'MAPE':
            self.metric = self.mape
        elif self.metric == 'R2':
            self.metric = self.r2

    def first_pred(self, y):
        if self.loss == 'MSE':
            return np.mean(y)
        elif self.loss == 'MAE':
            return np.median(y)
        else:
            raise Exception('Нет такого лосса')
    
    def change_leafs_values(self, y, leafs):
        for leaf in leafs:
            leaf_obj = np.array(y.loc[leaf.value[2]])
            leaf_pred = self.sum_preds[leaf.value[2]]
            new_value_leaf = leaf_obj - leaf_pred
            leaf.value[1] = self.first_pred(new_value_leaf) + self.reg * self.count_all_leafes

    def mae(self, y, y_pred):
        return sum(np.abs(y_pred - y)) / len(y)
    
    def mse(self, y, y_pred):
        return sum((y_pred - y) ** 2) / len(y)
    
    def rmse(self, y, y_pred):
        return np.sqrt(sum((y_pred - y) ** 2) / len(y))
    
    def mape(self, y, y_pred):
        return (100 / len(y)) * sum(np.abs((y_pred - y) / y))
    
    def r2(self, y, y_pred):
        return 1 - (sum((y_pred - y) ** 2)) / (sum((y - np.mean(y))**2))
    
    def dynamic_learning_rate(self):
        if isinstance(self.learning_rate, float):
            value = self.learning_rate
            self.learning_rate = lambda i: value

    def result_feature_importance(self):
        fi_new = pd.DataFrame([tree.fi for tree in self.trees]).sum().to_dict()
        fi_new = {key: value / 5 for key, value in fi_new.items()} # почему-то в курсе надо было в 2 раза меньше, либо ошибка в коде
        self.fi.update(fi_new)
