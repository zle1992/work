
import numpy as np
from numpy import float64
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import warnings
import pickle
import numpy as np  #惯例
import scipy as sp  #惯例
import pylab as pl  #惯例
from scipy.optimize import leastsq #这里就是我们要使用的最小二乘的函数


##################################拟合###################################
def fake_func(p, x):#定义拟合函数
    f = np.poly1d(p) #多项式分布的函数
    return f(x)
#残差函数
def residuals(p, y, x):     #残差函数
    regularization=0.5#正则化系数
    ret = y - fake_func(p, x)
    ret = np.append(ret, np.sqrt(regularization)*p) #将lambda^(1/2)p加在了返回的array的后面
    return ret 


def fit(sale_original):
    m = 14 #多项式的次数
    x =range(0,len(sale_original))#x 是0，1,2,3。。。。73
    y0 = sale_original#y0是销售量
    y1 = [np.random.normal(0, 10) + y for y in y0]#加入正态分布噪音后的y
    p0 = np.random.randn(m)#先随机产生一组多项式分布的参数
    plsq = leastsq(residuals, p0, args=(y0, x))#最小二乘法
    #print ('Fitting Parameters ：', plsq[0]) #输出拟合参数
    #画图形
    pl.plot(x, sale_original, label='real')
    pl.plot(x, fake_func(plsq[0], x), label='fitted curve')
    pl.legend()
    pl.show()
    return fake_func(plsq[0], x)

warnings.filterwarnings("ignore")#忽略错误警告
#读入文件
def read_old_data(file,area):    
    id_list=[]  #产品名字
    area_list=[]#某地区的销量
    data=[]    #产品属性  如功能 颜色等
    with open(file, 'r',encoding='gbk') as f: #打开文件
        for line in f:     #对每一行进行操作
            single_line = line.split(';')   #以逗号将将文件的第一行隔开
            id_list.append(single_line.pop(0))#删除产品名称
            area_list.append(single_line[area])  #地区销量
            data.append((single_line))    
    #print(data)
    for x in range(1,len(area_list)): #将销量转化为int  原始读入的是str类型
        area_list[x]=int(area_list[x])
    print("Modell")
    print(id_list[1:])
    return id_list[1:] ,area_list[1:],data[1:]  #返回数据，从第二列开始 因为第一列是属性名字
def read_new_data(file):    
    id_list=[]  #产品名字
    data=[]    #产品属性  如功能 颜色等
    with open(file, 'r',encoding='gbk') as f: #打开文件
        for line in f:     #对每一行进行操作
            single_line = line.split(';')   #以逗号将将文件的第一行隔开
            id_list.append(single_line.pop(0))#删除产品名称
            data.append((single_line))    
    #print(data)
    print("新产品名称")
    print(id_list[1:])
    return id_list[1:] ,data[1:]  #返回数据，从第二列开始 因为第一列是属性名字
        
    
    


#设置阈值将销量转化为需求 low high mid 
#input :销量列表   阈值1  阈值2
#output: 转化后的销量列表
def transform_sale(sale,threshold_1,threshold_2):
    sale_list=[]
    for x in range(0,len(sale)):
        if(int(sale[x])<threshold_1):
            sale_list.append('low')
        elif(int(sale[x])>threshold_2):
            sale_list.append('high')
        else:
            sale_list.append('mid')
    print("Ist-Bedarf：")
    print(sale_list)
    return sale_list


#数值化 处理标称属性
def preprocess(data,area):
    data=(np.asarray(data))#将list数据转化成矩阵
    le = preprocessing.LabelEncoder()     #将标称属性变成数字
    hot = preprocessing.OneHotEncoder(sparse=False)
    mm=preprocessing.MinMaxScaler(feature_range=(0, 1))#preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)：
#                                               将数据在缩放在固定区间，默认缩放到区间 [0, 1]
    #data=data[:,0:] #读取0到4列 对应 功能 速度 冷暖 颜色 价格
    #print(data)
    data_0=data[:,0]   #第一列
    data_0=le.fit_transform(data_0)#得到第一列转化后的结果
    for x in range(1,4): #range(1,4)=1,2,3 取剩下三列速度 冷暖 颜色
        data_single=data[:,x] 
        data_single=le.fit_transform(data_single)#特征矩阵进行定性特征编码的对象
        data_0=np.c_[data_0,data_single]#数据合并
    #data_0=hot.fit_transform(data_0)#数字转化成数字
    data_a=data[:,4].astype(float64)#数据只取价格
    data_a=mm.fit_transform(data_a)#价格处理
    #print(data_a)
    data_0=np.c_[data_0,data_a]  #合并功能与价格


    #print(data_0)
    return data_0

def split_data(data,tags,threshold,id_list):  
    train_data=data[:threshold,:]
    train_tags=tags[:threshold]
    test_data=data[threshold:,:]
    test_tags=tags[threshold:]
    return train_data, test_data, train_tags, test_tags ,id_list[threshold:]
def main(area,threshold_1,threshold_2):
    print('area:'+area)
    id_list,area_list,data=read_old_data(file,area_dict[area])#读取训练数据，返回矩阵
    
    print("area list trian：")
    print(area_list[:63])
    #arae_list=fit(area_list)#拟合图像
    sale_list=transform_sale(list(area_list),threshold_1,threshold_2)#将销量转化成需求高中低

    data=preprocess(data,area)#将标称数据转化成数字

    train_data, test_data, train_tags, test_tags= train_test_split( data,sale_list, test_size=0.3, random_state=1)#随机选择训练与测试数据
    #train_data, test_data, train_tags, test_tags,id_list_test=split_data( data,sale_list,63,id_list)#人为选择数据
#########################################  测试准确率
    from sklearn.svm import SVC
    clf = SVC(kernel = 'rbf',C=1)
    clf.fit(train_data,train_tags)  
    test_tags_pre = clf.predict(test_data)
    print("orign:")
    print(test_tags)
    print("predict:") 
    print(test_tags_pre)
    print ('accuracy_score:{0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
##############################################读取处理news数据
    new_id_list,new_data=read_new_data(new_file)#读取数据
    new_data=preprocess(new_data,area)  #预处理数据
##########################################################################预测new
  
    new_tags_pre= clf.predict(new_data) 
    print('new product:')
    print(new_tags_pre)

    

if __name__ == '__main__':
    file = 'data.csv'#文件名
    new_file='new.csv'
    area_dict={'hua_bei':5,'hua_dong':6,'hua_nan':7,'hua_zhong':8}#建立地区字典
    main('hua_bei',25,150 )#区域及阈值  
                                #'hua_bei',25,150  nbrs_single 0.7
                               #'hua_dong',25,500  nbrs_single   0.9
                               #'hua_nan',25,150 nbrs_single   0.7
                                #'hua_zhong',25,150  nbrs_single   0.7
                                #'all',100,3000     nbrs_single  0.9
    




