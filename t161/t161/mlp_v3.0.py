
import numpy as np
from numpy import float64
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import warnings
import matplotlib  
import matplotlib.pyplot as plt  
import sklearn.neural_network
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
def preprocess(data):
    data=(np.asarray(data))#将list数据转化成矩阵
    le = preprocessing.LabelEncoder()     #将标称属性变成数字
    hot = preprocessing.OneHotEncoder(sparse=False)
    mm=preprocessing.MinMaxScaler(feature_range=(0, 1))#preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)：
#                                               将数据在缩放在固定区间，默认缩放到区间 [0, 1]
    #data=data[:,0:5] #读取0到4列 对应 功能 速度 冷暖 颜色 价格
    #print(data.shape)
    data_0=data[:,0]   #第一列
    data_0=le.fit_transform(data_0)#得到第一列转化后的结果
    for x in range(1,4): #range(1,4)=1,2,3 取剩下三列速度 冷暖 颜色
        data_single=data[:,x] 
        data_single=le.fit_transform(data_single)#特征矩阵进行定性特征编码的对象
        data_0=np.c_[data_0,data_single]#数据合并
    #data_0=hot.fit_transform(data_0)#数字转化成数字
    data_a=data[:,4].astype(float64)#数据只取价格
    data_0=np.c_[data_0,data_a]  #合并功能与价格
    data_0=mm.fit_transform(data_0)#价格处理
    return data_0
def preprocess_new_data(data):
    data=(np.asarray(data))#将list数据转化成矩阵
    le = preprocessing.LabelEncoder()     #将标称属性变成数字
    hot = preprocessing.OneHotEncoder(sparse=False)
    mm=preprocessing.MinMaxScaler(feature_range=(0, 1))#preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)：
#                                               将数据在缩放在固定区间，默认缩放到区间 [0, 1]
    #data=data[:,0:5] #读取0到4列 对应 功能 速度 冷暖 颜色 价格
    #print(data)
    data_0=data[:,0]   #第一列
    data_0=le.fit_transform(data_0)#得到第一列转化后的结果
    for x in range(1,4): #range(1,4)=1,2,3 取剩下三列速度 冷暖 颜色
        data_single=data[:,x] 
        data_single=le.fit_transform(data_single)#特征矩阵进行定性特征编码的对象
        data_0=np.c_[data_0,data_single]#数据合并
    data_0=hot.fit_transform(data_0)#数字转化成数字
    data_a=data[:,4].astype(float64)#数据只取价格
    data_a=mm.fit_transform(data_a)#价格处理
    
    data_0=np.c_[data_0,data_a]  #合并功能与价格


    #print(data_0)
    return data_0
def split_data(data,tags,threshold,id_list):  
    train_data=data[:,:]
    train_tags=tags[:]
    test_data=data[threshold:,:]
    test_tags=tags[threshold:]
    return train_data, test_data, train_tags, test_tags ,id_list[threshold:]

# SVM Classifier using cross validation    
def svm_cross_validation(train_x, train_y):    
    from sklearn.grid_search import GridSearchCV    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [ 1, 4, 10, 100],   'kernel':['rbf', 'linear', 'poly','sigmoid'],   'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 10, verbose=1,cv=2)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
       print(para, val)    
       pass
    #print(grid_search.grid_scores_)
    print(grid_search.best_score_)
    model = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model 
def svm_plot(train_data,train_tags,test_data, test_tags):
    from sklearn.neighbors import KNeighborsClassifier  
    #clf = KNeighborsClassifier( n_neighbors=5)#default with k=5
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf = MLPClassifier(hidden_layer_sizes=(50,),  
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1,max_iter=1, alpha=0.4)
    #clf = AdaBoostClassifier(n_estimators=5)
    list_accuracy_score=[]
    list_parameter_1=[1,5,10,30,100,400]#
    list_parameter_2=[1e-4,1e-2,1e-1,1,2,4,10]
    #################################################
    for parameter_1 in list_parameter_1:
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
    best_parameter_1=list_parameter_1 [list_accuracy_score.index(max(list_accuracy_score))]
    print('best parameter c: {0:.3f}'.format(best_parameter_1))
    ######################################
    plt.plot( list_parameter_1,list_accuracy_score, 'bo-')
    plt.xticks(list_parameter_1, list_parameter_1, rotation=0)  
    plt.ylim([0.4,1.0])
    plt.title('mlp')
    plt.xlabel("max_iter")
    plt.ylabel("accuracy")
    plt.show()
    ###################################################
    #################################################
    list_accuracy_score=[]
    for parameter_2 in list_parameter_2:
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
    best_parameter_2=list_parameter_2 [list_accuracy_score.index(max(list_accuracy_score))]
    print('best parameter c: {0:.3f}'.format(best_parameter_2))
    ######################################
    plt.plot(list_parameter_2,list_accuracy_score, 'bo-')
    plt.xticks(list_parameter_2, list_parameter_2, rotation=0)  
    plt.ylim([0.5,1.0])
    plt.title('mlp')
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    plt.show()
    #return  KNeighborsClassifier( n_neighbors=best_parameter_1)
    #return MLPClassifier(alpha=best_parameter_1)
    return   MLPClassifier( hidden_layer_sizes=(50,),  
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1,max_iter=best_parameter_1, alpha=best_parameter_2)
def plot(list_x,list_y,x):
   # plt.subplot(2, 1, 1)
    plt.plot( list_x,list_y, 'bo-')
    plt.xticks(list_x, list_x, rotation=0)  
    plt.ylim([0.5,1.0])
    plt.title('SVM')
    plt.xlabel(x)
    plt.ylabel("accuracy")
    

    # plt.subplot(2, 1, 2)
    # plt.plot(x, y, 'r.-')
    # plt.ylim([0.5,1.0])
    # plt.xlabel('time (s)')
    # plt.ylabel('Undamped')

    plt.show()
    # ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax1.plot(x,y, "s-",linewidth=2,color="red")  
    # ax1.set_ylabel("list_accuracy_score")
    # ax1.set_xlabel("list_parameter_1")
    # ax1.set_ylim([0.5, 1.0])
    # ax1.legend(loc="lower right")
    # ax1.set_title('list_accuracy_score with parameter')
    # plt.show() #保存图  



def  main(area,threshold_1,threshold_2):
    print('area:'+area)
    id_list,area_list,data=read_old_data(file,area_dict[area])#读取训练数据，返回矩阵 
    print("area list trian：")
    print(area_list[:63])
    sale_list=transform_sale(list(area_list),threshold_1,threshold_2)#将销量转化成需求高中低
    data=preprocess(data)#将标称数据转化成数字
    train_data, test_data, train_tags, test_tags= train_test_split( data,sale_list, test_size=0.25, random_state=1)#随机选择训练与测试数据
    #train_data, test_data, train_tags, test_tags,id_list_test=split_data( data,sale_list,63,id_list)#人为选择数据
    #clf=svm_cross_validation(train_data, train_tags)
#########################################  测试准确率
    # # # from sklearn.svm import SVC

    
    clf=svm_plot(train_data,train_tags,test_data, test_tags)
    clf.fit(train_data,train_tags)
    test_tags_pre = clf.predict(test_data)
    print("orign:")
    print(test_tags)
    print("predict:") 
    print(test_tags_pre)
    print ('accuracy_score:{0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
#############################################读取处理news数据
    new_id_list,new_data=read_new_data(new_file)#读取数据
    new_data=preprocess(new_data)  #预处理数据
# ##########################################################################预测new
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
    




