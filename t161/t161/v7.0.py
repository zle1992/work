
import numpy as np
from numpy import float64
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier  
import warnings
import matplotlib  
import matplotlib.pyplot as plt  
import time
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
    #print("Modell")
    #print(id_list[1:])
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
    #print("Ist-Bedarf：")
    #print(sale_list)
    return sale_list




#数值化 处理标称属性
def preprocess(data):
    data=(np.asarray(data))#将list数据转化成矩阵
    le = preprocessing.LabelEncoder()     #将标称属性变成数字
    mm=preprocessing.MinMaxScaler(feature_range=(0, 1))#preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)：
#                                               将数据在缩放在固定区间，默认缩放到区间 [0, 1]
    data_0=data[:,0]   #第一列
    data_0=le.fit_transform(data_0)#得到第一列转化后的结果
    for x in range(1,4): #range(1,4)=1,2,3 取剩下三列速度2 冷暖2 颜色3
        data_single=data[:,x] 
        data_single=le.fit_transform(data_single)#特征矩阵进行定性特征编码的对象
        data_0=np.c_[data_0,data_single]#数据合并
    data_a=data[:,4].astype(float64)#数据只取价格
    data_0=np.c_[1*data_0,data_a]  #合并功能与价格
    data_0=mm.fit_transform(data_0)#价格处理
    return data_0



def split_data(data,tags,threshold,id_list):  
    train_data=data[:threshold,:]
    train_tags=tags[:threshold]
    test_data=data[threshold:,:]
    test_tags=tags[threshold:]
    return train_data, test_data, train_tags, test_tags ,id_list[threshold:]



def plot(list_parameter_1,list_accuracy_score,title,xlabel,ylabel):
    plt.plot(range(0,len(list_parameter_1)),list_accuracy_score, 'bo-')
    plt.xticks(range(0,len(list_parameter_1)), list_parameter_1, rotation=0)  
    plt.ylim([0.3,1.0])
    plt.title(title) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

 


def svm_plot(train_data,train_tags,test_data, test_tags):
    list_accuracy_score=[]
    print("svm ")
    list_parameter_1=['rbf', 'linear','poly','sigmoid']# 核函数
    list_parameter_2=range(1,100)#C  你可以修改
    #################################################
    for parameter_1 in list_parameter_1:
        clf = SVC(kernel = parameter_1,C=10)
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
    best_parameter_1=list_parameter_1 [list_accuracy_score.index(max(list_accuracy_score))]
    print("best parameter1: "+(best_parameter_1))
    ######################################
    plot(list_parameter_1,list_accuracy_score,'SVM',"kernel","accuracy")
    ###################################################
    #################################################
    list_accuracy_score=[]
    for parameter_2 in list_parameter_2:
        clf = SVC(C = parameter_2)
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
    best_parameter_2=list_parameter_2 [list_accuracy_score.index(max(list_accuracy_score))]
    print('best parameter2 : {0:.3f}'.format(best_parameter_2))
    ######################################
    #plot(list_parameter_2,list_accuracy_score,'SVM',"c","accuracy")
    return SVC(kernel=best_parameter_1, C = best_parameter_2)
def rf_plot(train_data,train_tags,test_data, test_tags):
    from sklearn.ensemble import RandomForestClassifier 
    print("rf ")
    list_accuracy_score=[]
    list_oob_error=[]
    list_parameter_1=range(1,50)#
    list_parameter_2=range(1,50)#
    list_parameter_3=range(1,5)#
    #################################################
    for parameter_1 in list_parameter_1:
        clf = RandomForestClassifier(max_depth=parameter_1, n_estimators=10, max_features=1,oob_score=True)
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
        list_oob_error.append(1 - clf.oob_score_)
    best_parameter_1=list_parameter_1 [list_accuracy_score.index(max(list_accuracy_score))]
    print('best parameter1 : {0:.3f}'.format(best_parameter_1))
    ######################################
    # plot(list_parameter_1, list_oob_error,'rf',"max_depth","oob_error")
    # plot(list_parameter_1,list_accuracy_score,'rf',"max_depth","accuracy")
    ###################################################
    #################################################
    list_accuracy_score=[]
    list_oob_error=[]
    for parameter_2 in list_parameter_2:
        clf = RandomForestClassifier(max_depth=best_parameter_1, n_estimators=parameter_2, max_features=1,oob_score=True)
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
        list_oob_error.append(1 - clf.oob_score_)
    best_parameter_2=list_parameter_2 [list_accuracy_score.index(max(list_accuracy_score))]
    print('best parameter2 : {0:.3f}'.format(best_parameter_2))
    ######################################
    # plot(list_parameter_2, list_oob_error,'rf',"n_estimators","oob_error")
    # plot(list_parameter_2,list_accuracy_score,'rf',"n_estimators","accuracy")
    list_accuracy_score=[]
    list_oob_error=[]
    for parameter_3 in list_parameter_3:
        clf = RandomForestClassifier(max_depth=best_parameter_1, n_estimators=best_parameter_2, max_features=parameter_3,oob_score=True)
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
        list_oob_error.append(1 - clf.oob_score_)
    best_parameter_3=list_parameter_3 [list_accuracy_score.index(max(list_accuracy_score))]
    print('best parameter3 : {0:.3f}'.format(best_parameter_3))
    ######################################
    # plot(list_parameter_3, list_oob_error,'rf',"max_features","oob_error")
    # plot(list_parameter_3,list_accuracy_score,'rf',"max_features","accuracy")
    return  RandomForestClassifier(max_depth=best_parameter_1, n_estimators=best_parameter_2, max_features=best_parameter_3) 
def knn_plot(train_data,train_tags,test_data, test_tags):
    list_accuracy_score=[]
    print("knn")
    list_parameter_1=range(1,20)#
    #################################################
    for parameter_1 in list_parameter_1:
        clf = KNeighborsClassifier( n_neighbors=parameter_1)
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
    best_parameter_1=list_parameter_1 [list_accuracy_score.index(max(list_accuracy_score))]
    print('best parameter : {0:.3f}'.format(best_parameter_1))
    ######################################
    #plot(list_parameter_1,list_accuracy_score,'knn',"k","accuracy")
    ###################################################
   
    return  KNeighborsClassifier( n_neighbors=best_parameter_1)
 
def ada_plot(train_data,train_tags,test_data, test_tags):
    #
    print("ada ")
    list_accuracy_score=[]
    list_parameter_1=range(1,50)
    #################################################
    for parameter_1 in list_parameter_1:
        clf = AdaBoostClassifier(n_estimators=parameter_1)
        clf.fit(train_data,train_tags)  
        test_tags_pre = clf.predict(test_data)
        list_accuracy_score.append(' {0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
    best_parameter_1=list_parameter_1 [list_accuracy_score.index(max(list_accuracy_score))]
    print('best parameter : {0:.3f}'.format(best_parameter_1))
    ######################################
    # plot(list_parameter_1,list_accuracy_score,'ada',"n_estimators","accuracy")
    ###################################################
    return   AdaBoostClassifier(n_estimators=best_parameter_1 )

def  main(area,threshold_1,threshold_2):

    #print('area:'+area)
    id_list,area_list,data=read_old_data(file,area_dict[area])#读取训练数据，返回矩阵 
    #print("area list trian：")
    #print(area_list[:63])
    sale_list=transform_sale(list(area_list),threshold_1,threshold_2)#将销量转化成需求高中低
    data=preprocess(data)#将标称数据转化成数字
    #train_data, test_data, train_tags, test_tags= train_test_split( data,sale_list, test_size=0.25, random_state=1)#随机选择训练与测试数据
    train_data, test_data, train_tags, test_tags,id_list_test=split_data( data,sale_list,63,id_list)#人为选择数据

#########################################  选择分类器
    #clf=rf_plot(train_data,train_tags,test_data, test_tags)
    #clf=svm_plot(train_data,train_tags,test_data, test_tags)
    #clf=knn_plot(train_data,train_tags,test_data, test_tags)
    clf=ada_plot(train_data,train_tags,test_data, test_tags)
######################################### 
    clf.fit(train_data,train_tags)
    test_tags_pre = clf.predict(test_data)

    # print("orign:")
    # print(test_tags)
    # print("predict:") 
    # print(test_tags_pre)

    print ('accuracy_score:{0:.3f}'.format(accuracy_score(test_tags, test_tags_pre)))
#############################################读取处理news数据
    # new_id_list,new_data=read_new_data(new_file)#读取数据
    # new_data=preprocess(new_data)  #预处理数据
# ##########################################################################预测new
    # new_tags_pre= clf.predict(new_data) 
    # print('new product:')
    # print(new_tags_pre)


    return accuracy_score(test_tags, test_tags_pre)
def find_best():
    threshold_1=range(30,40,1)# 10是间隔   1，11.。。。。
    threshold_2=range(100,120,1)
    list_accuracy_score=[]
    for x in threshold_1:
        for y in threshold_2:
            print("x :",x,"y: ",y)
            list_accuracy_score.append(main('hua_bei',x,y))
    best_index=list_accuracy_score.index(max(list_accuracy_score))
    best_y=best_index%len(threshold_2)
    best_x=int((best_index-best_y)/len(threshold_2))
    print("best x " ,threshold_1[best_x],"best y :",threshold_2[best_y])
    print('best_accuracy_score:{0:.3f}'.format(max(list_accuracy_score)))

if __name__ == '__main__':
    
    start=time.time()
    file = 'data.csv'#文件名
    new_file='new.csv'
    area_dict={'hua_bei':5,'hua_dong':6,'hua_nan':7,'hua_zhong':8}#建立地区字典
    #find_best()
    main('hua_bei',36,112)#区域及阈值  
                                #'hua_bei',36,112  nbrs_single 0.7
                               #'hua_dong',91,535  nbrs_single   0.9
                               #'hua_nan',69,236 nbrs_single   0.7
                                #'hua_zhong',45,130  nbrs_single   0.7
                                #'all',100,3000     nbrs_single  0.9
    print("runtime: {0:.5f}".format(time.time()-start)+'s')
    


#2017 14:30    开始
#2017 15:25     简化程序  finash!
#2017 17:05    oob error  finish !
#2017 19:36   find_best threshold finish! 
#2017 21:30    plot finish! 

#2017 1.18 11:40    开始
#2017 1.18 12:00    runtime finish! 运行时间太短

#2017 1.18 18:10    开始
#2017 1.18 18:55    精简程序
#2017 1.18 19:41    fix find best
#2017 1.18 19:50    fix the split method