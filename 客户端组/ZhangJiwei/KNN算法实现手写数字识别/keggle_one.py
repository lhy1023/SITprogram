
"""
1.预处理
读取训练数据集，拆分为trainlabel 和traindata两个矩阵，同时数值化和归一化
"""

train = pd.read_csv("train.csv")
    trainlabel = ravel(toint(train.iloc[:, 0]).transpose())
    traindata = toint(train.iloc[:, 1:])
 
    test = pd.read_csv("test.csv")
    testdata = toint(test.values)
 
    # 对traindata和testdata归一化，将所有不为0的数值全部转换为1
    train_rows = traindata.shape[0]
    train_columns = traindata.shape[1]
    test_rows = testdata.shape[0]
    test_columns = testdata.shape[1]
 
    traindata = nomalizing(traindata, train_rows, train_columns)
    testdata = nomalizing(testdata, test_rows, test_columns)



"""数值化是指原数据集的数据类型可能是字符串，由于后面会做是否为0的判断，所以需要将数据类型转换为数值型"""
def toint(array):
    """转为数值型数据"""
    array = mat(array)# 生成矩阵
    m, n = shape(array)
    newArray = zeros((m, n))
    for i in range(m):
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray


"""归一化是指将所有不为0的数都转变为1，目的是简化操作"""
def nomalizing(data,r,l):
    """一个标签对应784个特征，特征值为0-255的灰度值，0代表黑色，255代表白色，此处将所有不为0的值都以1转换，简化工作量"""
    for i in range(r):
        for j in range(l):
            if data[i,j] != 0:
                data[i,j] = 1
    return data



"""
2. 交叉检验
使用交叉检验，目的是评估n_neighbors参数的最佳取值，减少过拟合。
因为此处只控制一个参数k,即n_neighbors参数，所以用模型选择库中的cross_val_score(),若有多个参数则需使用GridSearchCV()
"""
scores_list = []
k_range = range(1,10)
for k in k_range:
    knnclf = KNeighborsClassifier(k)
    scores = cross_val_score(knnclf,traindata,trainlabel,cv=10,scoring="accuracy")
    scores_list.append(mean(scores))
plt.plot(k_range,scores_list)
plt.show()
print(max(scores_list))    # 最大精准度
k = argsort(array(scores_list))[-1]    # 最大精准度对应的k值
"""
此处将k值得范围锁定到1-9，cv=10设定为十折交叉验证，score ="accuracy"评分标准采用精确率，默认是以 scoring=’f1_macro’
取得k值和对应准确率的折线图，可得到最佳k值和最大准确率
"""


"""
3. 算法实现
lavel()的作用是将多维数组降为一维，由于label只有一列数值，每一行特征对应一个label,需要将shape转变为（42000，）的形式，所以必须降维
扩展：lavel()和flatten()有一样的效果，不过唯一区别是flatten返回的是副本，而如果改变lavel会改变原数据
"""
def knnClassify(k,data,label,test):
    """KNN算法"""
    knnclf = KNeighborsClassifier(k)
    knnclf.fit(data,ravel(label))    # label降维一维数据
    testlabel = knnclf.predict(test)
    save_result(testlabel,"testlabel.csv")
 
def save_result(data,filename):
    """保存预测结果"""
    # newline参数是控制文本模式之下，一行的结束字符
    with open(filename,'w',newline="") as f:
        w = csv.writer(f)
        for d in data:
            tmp = []
            tmp.append(d)
            w.writerow(tmp)

"""
K近邻算法KNN原理是：根据与待测试样本距离最近的k个训练样本的大概率分类来判断待测样本所归属的类别
过程是：
1）计算测试数据与各个训练数据之间的距离；
2）按照距离的递增关系进行排序；
3）选取距离最小的K个点；
4）确定前K个点所在类别的出现频率；
5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。
"""
