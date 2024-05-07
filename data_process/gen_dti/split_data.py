

from sklearn.model_selection import KFold
import numpy as np

all=np.genfromtxt('label.txt')
all = np.arange(0, len(all), 1)
kf = KFold(n_splits=5,shuffle=True,random_state=42)  # 初始化KFold
train=[]
test=[]
for train_index , test_index in kf.split(all):  # 调用split方法切分数据
    # print('train_index:%s , test_index: %s ' %(train_index,test_index))
    train.append(train_index)
    test.append(test_index)
for i in range(5):
    testN="test00"+str(i+1)+".txt"
    trainN="train00"+str(i+1)+".txt"
    np.savetxt(testN, np.array(test[i]), fmt="%d")
    np.savetxt(trainN, np.array(train[i]), fmt="%d")

