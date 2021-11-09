# 邮件分类 垃圾邮件识别

## 核心类库
* scikit-learn 机器学习专用三方库
* pickle 模型保存

## 目录结构
---
---

###########目录结构描述
````

├── data                        // 训练数据
├── models                      // 模型保存目录
│   ├── Bayes.pkl               // 贝叶斯算法模型
│   ├── semodel.pickle          // 词频选取模型
│   ├── vmodel.pickle           // TF-idf词频模型
├── test                        // 序列化索引数据
├── email_preprocess.py         // 预处理模块 
├── main.py                     // 内容识别
├── train.py                    // 算法模型训练
├── README.md                    
└── requirements.txt      

````


## 模型训练

````python
#运行

python train.py

#生产使用训练模型
````
#### 模型训练：

````python

clf = GaussianNB()#贝叶斯算法选取

    t0 = time()
    clf.fit(features_train, labels_train)

#以下算法选择供选择
clf = SVC(kernel = 'linear', C=1)#向量机
clf = tree.DecisionTreeClassifier()#决策树
clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')#KNN近邻算法

clf = RandomForestClassifier(max_depth=2, random_state=0)#随机森林 深度50左右 识别率最好


````

##内容识别分类
````python
#运行

python main.py

#根据已经训练好的模型  对未知数据进行预测
````
