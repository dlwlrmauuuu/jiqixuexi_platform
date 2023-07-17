import csv
import math
import time

import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from flask_socketio import emit
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import Models as models

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)


class Logger:
    def __init__(self, name, trigger=None) -> None:
        self.name = name
        self.trigger = trigger

    def print(self, *x, **kwargs):
        if self.trigger is not None:
            self.trigger({
                'type': 'print',
                'name': self.name,
                'args': x,
                'kwargs': kwargs
            })
            return
        emit('log_message',
             {'type': 'print', 'name': self.name, 'message': ' '.join(map(str, x)), 'kwargs': kwargs})

    def imshow(self, img):
        if self.trigger is not None:
            self.trigger({
                'type': 'imshow',
                'name': self.name,
                'args': img
            })
            return
        emit('log_message', {'type': 'imshow', 'name': self.name, 'image': img})

    @classmethod
    def get_logger(cls, name):
        logger = cls(name, trigger=Logger.global_trigger)
        cls.loggers.setdefault(id(logger), logger)
        return logger


Logger.loggers = {}
Logger.global_trigger = None


# 通过文件读入数据集
class ReadDataset:
    def __init__(self, number) -> None:
        super().__init__()
        self.logger = Logger.get_logger('BasicDataSet')
        self.data = []
        self.count = 0
        filenames = ['static/DataSets/BostonHousing.csv', 'static/DataSets/Diabetes.csv',
                     'static/DataSets/Ionosphere.csv']
        if (number >= 0) and (number < len(filenames)):
            self.logger.print("Try to load dataset {}...".format(number + 1))
            with open(filenames[number], 'r') as csvfile:
                csvreader = csv.reader(csvfile, dialect='excel')
                for row in csvreader:
                    if csvreader.line_num == 1:
                        continue
                    self.data.append(list(map(float, row[1:])))
                    self.count += 1
        self.logger.print("Done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def len(self):
        return self.count


class TrainTestDataset:
    def __init__(self, item) -> None:
        super().__init__()
        self.item = item

    def __len__(self) -> int:
        return len(self.item)

    def __getitem__(self, idx: int):
        return self.item[idx]


class DivisionAndSplitter:
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger = Logger.get_logger('DivisionAndSplitter')
        # self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset):
        self.logger.print("Start splitting...")
        trainingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio)) if i != 0]
        trainingSet = TrainTestDataset(trainingSet)

        testingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio), len(dataset)) if i != 0]
        testingSet = TrainTestDataset(testingSet)

        # self.logger.print("split!")
        # self.logger.print("training_len = {}".format([trainingSet[i] for i in range(len(trainingSet))]))
        # self.logger.print("training_len = {}".format([testingSet[i] for i in range(len(testingSet))]))
        self.logger.print("Splitting completed.")
        return trainingSet, testingSet

    def division(self, training, testing):
        self.logger.print("Start dividing..")
        train_x = [row[:-1] for row in training]
        train_y = [row[-1] for row in training]
        test_x = [row[:-1] for row in testing]
        test_y = [row[-1] for row in testing]
        self.logger.print("Dividing completed.")
        # self.logger.print("y_train_len = {}".format([train_y[i] for i in range(len(train_y))]))
        return train_x, train_y, test_x, test_y


class TrainingModels():
    def __init__(self, x_train, y_train, x_test, y_test, number, parameter_list) -> None:
        super().__init__()
        self.time_end = None
        self.time_start = None
        self.y_pred = None
        self.models = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.flag = number
        self.parameter_list = parameter_list
        self.logger = Logger.get_logger('Models')

    def fit_predict(self, x_train, y_train, x_test):

        self.logger.print("Loading models...")

        if self.flag == 0:
            self.models = models.Linear()
            self.logger.print("Using Linear Regression...")
        elif self.flag == 1:
            if len(self.parameter_list) != 3:
                self.models = models.SVM(kernel='rbf', degree=3, tol=0.001)
            else:
                self.models = models.SVM(kernel=self.parameter_list[0], degree=int(float(self.parameter_list[1])),
                                         tol=float(self.parameter_list[2]))
            self.logger.print("Using SVM...")
        elif self.flag == 2:
            self.models = models.KNN()
            self.logger.print("Using KNN...")
        elif self.flag == 3:
            if len(self.parameter_list) != 2:
                self.models = models.Logistic(learning_rate=0.003, iterations=100)
            else:
                self.models = models.Logistic(learning_rate=float(self.parameter_list[0]),
                                              iterations=int(float(self.parameter_list[1])))
            self.logger.print("Using Logistic Regression...")
        elif self.flag == 4:
            if len(self.parameter_list) != 4:
                self.models = models.DecisionTreeClassifier(criterion='gini', max_depth=5, d=4,
                                                            random_state=0)
            # ['gini', 'entropy','error']
            else:
                self.models = models.DecisionTreeClassifier(criterion=self.parameter_list[0],
                                                            max_depth=int(float(self.parameter_list[1])),
                                                            d=int(float(self.parameter_list[2])),
                                                            random_state=int(float(self.parameter_list[3])))
            self.logger.print("Using Decision Tree...")
        elif self.flag == 5:
            self.models = models.KMeans()
            self.logger.print("Using K-Means...")
        elif self.flag == 6:
            if len(self.parameter_list) != 5:
                self.models = models.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=5,
                                                            d=4, random_state=0)
            else:
                self.models = models.RandomForestClassifier(n_estimators=int(float(self.parameter_list[0])),
                                                            criterion=self.parameter_list[1],
                                                            max_depth=int(float(self.parameter_list[2])),
                                                            d=int(float(self.parameter_list[3])),
                                                            random_state=int(float(self.parameter_list[4])))
            self.logger.print("Using Random Forest...")
        elif self.flag == 7:
            self.models = models.Beyes()
            self.logger.print("Using Naive Bayes...")
        elif self.flag == 8:
            if len(self.parameter_list) != 2:
                self.models = models.DimReduction(n_components=3, whiten=False)
            else:
                self.models = models.DimReduction(n_components=int(float(self.parameter_list[0])),
                                                  whiten=self.parameter_list[1])
            self.logger.print("Using Dimensional Reduction...")
        elif self.flag == 9:
            if len(self.parameter_list) != 4:
                self.models = models.GBDT(criterion='gini', max_depth=5, d=4,
                                          random_state=0)
            else:
                self.models = models.GBDT(criterion=self.parameter_list[0],
                                          max_depth=int(float(self.parameter_list[1])),
                                          d=int(float(self.parameter_list[2])),
                                          random_state=int(float(self.parameter_list[3])))
            self.logger.print("Using Gradient Boosting...")
        else:
            if len(self.parameter_list) != 4:
                self.models = models.Graboosting(flag=0, eta=0.3, max_depth=6, subsample=1)
            else:
                self.models = models.Graboosting(flag=int(float(self.parameter_list[0])),
                                                 eta=float(self.parameter_list[1]),
                                                 max_depth=int(float(self.parameter_list[2])),
                                                 subsample=float(self.parameter_list[3]))
            self.logger.print("Using eXtreme Gradient Boosting...")

        self.logger.print("Fitting models...")
        self.time_start = time.time()
        self.models.fit(x_train, y_train)
        self.logger.print("Loading And Fitting completed.")
        self.logger.print("Start forecasting...")
        self.y_pred = self.models.predict(x_test)
        self.time_end = time.time()
        self.logger.print("Forecasting completed.")
        self.logger.print("Total Consume time : {:.3f} ms ".format((self.time_end - self.time_start) * 1000))
        # self.logger.print("y_pred_len = {}".format([self.y_pred[i] for i in range(len(self.y_pred))]))

        return self.y_pred


class EvaluationStandard():
    def __init__(self, y_test, y_pred, code_list) -> None:
        super().__init__()
        self.y_test = y_test
        self.y_pred = y_pred
        self.code_list = code_list
        self.logger = Logger.get_logger('Standard')
        self.logger2 = Logger.get_logger("Error")
        if all(y % 1 == 0 for y in y_test) and all(y % 1 == 0 for y in y_pred):
            self.logger.print("Use classification")
            # 分类问题基本指标
            cm = confusion_matrix(y_test, y_pred)
            cm = np.array(cm)
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            self.FP = FP.astype(float)
            self.FN = FN.astype(float)
            self.TP = TP.astype(float)
            self.TN = TN.astype(float)
        else:
            # 回归问题基本指标
            self.logger.print("Use regression")
            self.yt = np.array(y_test)
            self.yp = np.array(y_pred)
            self.sse = np.sum((self.yt - self.yp) ** 2)
            self.ssr = np.sum((np.mean(self.yt) - self.yp) ** 2)
            self.sst = np.sum((self.yp - np.mean(self.yp)) ** 2)

    # 均方误差MSE
    def MSE_rate(self):
        mse = metrics.mean_squared_error(self.y_test, self.y_pred)
        return mse

    # 错误率
    def wrong_rate(self):
        rate = (self.FN + self.FP) / len(self.y_test)
        return rate[0]

    # 精度
    def acc_rate(self):
        rate = (self.TN + self.TP) / len(self.y_test)
        return rate[0]

    # 查准率
    def precision_rate(self):
        rate = self.TP / (self.FP + self.TP)
        return rate[0]

    # 查全率
    def recall_rate(self):
        rate = self.TP / (self.FN + self.TP)
        return rate[0]

    # F1指数
    def F1_rate(self):
        rate = 2 * self.TP / (len(self.y_test) - self.TN + self.TP)
        return rate[0]

    # R2判定指数
    def R2_rate(self):
        rate = 1 - self.sse / self.sst
        return rate

    # 平均绝对误差
    def MAE_rate(self):
        rate = 0
        for i in range(len(self.yt)):
            rate += math.fabs(self.yt[i] - self.yp[i])
        rate = rate / len(self.yt)
        return rate

    def check(self, nums):
        one_seven_eight = ['1', '7', '8']
        two_three_four = ['2', '3', '4','5', '6']

        if set(nums).issubset(one_seven_eight) or set(nums).issubset(two_three_four):
            return True
        else:
            return False

    def ReturnResults(self):
        answer = []
        if self.check(self.code_list):
            for i in self.code_list:
                if i == '1':
                    answer.append('mse_rate: ' + str(self.MSE_rate()))
                elif i == '2':
                    answer.append('wrong_rate:' + str(self.wrong_rate()))
                elif i == '3':
                    answer.append('acc_rate:' + str(self.acc_rate()))
                elif i == '4':
                    answer.append('precision:' + str(self.precision_rate()))
                elif i == '5':
                    answer.append('recall_rate:' + str(self.recall_rate()))
                elif i == '6':
                    answer.append('F1_rate:' + str(self.F1_rate()))
                elif i == '7':
                    answer.append('R2_rate:' + str(self.R2_rate()))
                else:
                    answer.append('MAE_rate:' + str(self.MAE_rate()))
            return answer
        else:
            self.logger2.print("Wrong evaluation index has been selected. Please check.")


def func(dataset_no, rate, model_no, para_list):
    dataset = ReadDataset(dataset_no)
    splitter = DivisionAndSplitter(rate)
    train_data_test_data = splitter.split(dataset)
    train_data, test_data = train_data_test_data[0], train_data_test_data[1]
    temp = splitter.division(train_data, test_data)
    x_train, y_train, x_test, y_test = temp[0], temp[1], temp[2], temp[3]
    model = TrainingModels(x_train, y_train, x_test, y_test, model_no, para_list)
    y_pred = model.fit_predict(x_train, y_train, x_test)
    return y_test, y_pred


def get_moxing(data):
    if data['moxing'] == "线性回归":
        return 0
    elif data['moxing'] == "支持向量机":
        return 1
    elif data['moxing'] == "K-近邻":
        return 2
    elif data['moxing'] == "逻辑回归":
        return 3
    elif data['moxing'] == "决策树":
        return 4
    elif data['moxing'] == "K平均":
        return 5
    elif data['moxing'] == "随机森林":
        return 6
    elif data['moxing'] == "朴素贝叶斯":
        return 7
    elif data['moxing'] == "降维算法":
        return 8
    elif data['moxing'] == "GBDT":
        return 9
    elif data['moxing'] == "梯度增强":
        return 10


@socketio.on('client_message')
def handle_client_message(data):
    t_list = func(data['shujuji'], float(data['fengebili']), get_moxing(data), data['private_para'])
    outcome = str(t_list[1])
    print(data)
    print(outcome)
    # print(outcome.shape[0])
    # for i in range()
    ss='null'
    if(data['moxing'] != '降维算法'):
        juede = EvaluationStandard(t_list[0], t_list[1], data['pingjiazhibiao'])
        ss =str(juede.ReturnResults())
        print(ss)

    emit('server_message', {'outcome': outcome, 'evaluation_standard': ss})


# 第一个数据集划分方法和算法
hf1 = ["随机"]
mx1 = {"线性回归", "梯度增强"}
# 第二个数据集划分方法和模型
hf2 = ["随机"]
mx2 = ["支持向量机", "K-近邻", "逻辑回归", "决策树", "随机森林", "朴素贝叶斯", "降维算法", "梯度增强", "GBDT"]
# 第三个数据集划分方法和模型
hf3 = ["随机"]
mx3 = ["支持向量机", "K-近邻", "逻辑回归", "决策树", "随机森林", "朴素贝叶斯", "降维算法", "梯度增强", "GBDT"]


def generate_divs(selected_option):
    # 模拟根据选中的算法返回需要的参数列表
    divs = []
    if selected_option == "线性回归":
        divs = []
    elif selected_option == "支持向量机":
        divs = ["核函数", "多项式核阶数", "残差收敛条件"]
    elif selected_option == "K-近邻":
        divs = ["k"]
    elif selected_option == "逻辑回归":
        divs = ["学习率", "迭代次数"]
    elif selected_option == "决策树":
        divs = ["特征选取方法", "最大深度", "选择特征数", "随机数种子"]
    elif selected_option == "K平均":
        divs = ["聚类数", "生成初始质心方法", "迭代次数"]
    elif selected_option == "随机森林":
        divs = ["树的数量", "特征选取方法", "最大深度", "选择特征数", "随机数种子"]
    elif selected_option == "朴素贝叶斯":
        divs = []
    elif selected_option == "降维算法":
        divs = ["降维后维度", "白化"]
    elif selected_option == "梯度增强":
        divs = ["分类或回归", "学习率", "最大深度", "抽样比例"]
    elif selected_option == "GBDT":
        divs = ['特征选取方法', '最大深度', '选择特征数', '随机数种子']

    return divs


# 生成动态参数
@app.route("/get_divs", methods=["POST"])
def get_divs():
    selected_option = request.form.get("option")
    print(selected_option)
    divs = generate_divs(selected_option)
    # print(divs)
    return jsonify(divs)


@app.route('/')
def page():
    return render_template('index.html')


@app.route('/1.html')
def page1():
    return render_template('1.html', hf=hf1, mx=mx1)


@app.route('/2.html')
def page2():
    return render_template('2.html', hf=hf2, mx=mx2)


@app.route('/3.html')
def page3():
    return render_template('3.html', hf=hf3, mx=mx3)


if __name__ == '__main__':
    socketio.run(app, host='localhost', port=5000)
