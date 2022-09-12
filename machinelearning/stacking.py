import numpy as np
import pandas as pd
from keras import layers
from keras import models
"""
The following project is very simple. Implement only the locations mentioned.
Your job is to implement basic algorithms. To make it more readable, you can implement
the algorithms in another file and call here. The results are important...
"""

# Path to data and read data
PATH_TO_DATA = 'S Tejarat.Bank.csv'
TSEtmc = pd.read_csv(PATH_TO_DATA)
df = TSEtmc[['<CLOSE>', '<OPEN>', '<LOW>','<HIGH>']]
df = df.iloc[::-1].reset_index(drop=True)

# Make Label
next_day_label = np.sign(df['<CLOSE>'].diff(1).shift(-1).values)
next_day_label[np.where(next_day_label == 0)] = 1
df['Y'] = next_day_label
df.drop(df.tail(1).index, inplace=True)
def acc(x,theta,y):
    return sum(predict(x,theta)==y)/len(y)

# function for separating test and train data 0.2,0.8
def test_train(dataset):
    dataset_train = dataset[:int((len(dataset) * 0.8))]
    dataset_test = dataset[int((len(dataset) * 0.8)):len(dataset)]
    y_train = dataset_train['Y']
    x_train = dataset_train.drop(['Y'], axis=1)
    y_test = dataset_test['Y']
    x_test = dataset_test.drop(['Y'], axis=1)
    return y_train, x_train, y_test, x_test
y_train, x_train, y_test, x_test = test_train(df)
x_train=x_train.to_numpy()
x_test=x_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
#-------------------------------------
# Implement the function of the first algorithm below:
''' svm with kernel'''
def matrisegau(X,sigma):
    row=X.shape[0]
    col=X.shape[1]
    matrisegau=np.zeros(shape=(row,row))
    i=0
    for v_i in X:
        j=0
        for v_j in X:
            matrisegau[i,j]=Gaussian(v_i.T,v_j.T,sigma)
            j+=1
        i+=1
    return matrisegau
def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))
def hyp(x,theta,b):
    d=np.dot(x,theta)+b
    return d
def pre(x,theta,b):
    d=np.dot(x,theta)+b
    d[d>=0]=1
    d[d<0]=-1
    return d
def cost(w, x, y,c,b):
    j1=np.dot(w.transpose(),w)/2
    j2=0
    for i in range (len(y)):
        h=hyp(x,w,b)
        j2=j2+np.maximum(0,1-y[i]*(h[i]))
    j=j1+j2*c
    return j
def grad3(w,x,y,c,b,alpha):
    dw=np.zeros((len(w),1))
    db=0
    h=hyp(x,w,b)
    for i in range (len(y)):
        if h[i]*y[i]<=1:
            db+=-y[i]
            for j in range (len(w)):
                dw[j]+=-y[i]*x[i,j]
            
    dw=c*dw
    dj=w
    djw = dw + dj
    dwb = c*alpha*db
    w=w-alpha*djw
    b = b - dwb
    return w,b

def algorithm_1(x_train, y_train,x_test):
    
    x_train=matrisegau(x_train,0.3)
    x_test=matrisegau(x_test,0.3)
    theta=np.zeros(len(x_train))
    theta=np.expand_dims(theta, axis=1)
    b=1
    for i in range (100): 
        theta,b=grad3(theta,x_train,y_train,0.1,b,0.001)
    a=pre(x_test,theta,b)
    return a
#-------------------------------------
# Implement the function of the second algorithm below:
def algorithm_2(x_train, y_train,x_test):
    pre=np.zeros(len(y_test))
    k=7
    for i in range(len(x_test)):
            dis=[]
            for j in range(len(x_train)):
                dis.append(np.linalg.norm(x_test[i]-x_train[j]))
            sortindex=np.argsort(dis)
            l=0
            for m in range (k):
               if y_train[sortindex[m]]==1:
                   l+=1
               else:
                   l-=1
            if l>0:
                pre[i]=1
            else:
                pre[i]=0
    return pre
#-------------------------------------
# Implement the function of the third algorithm below:
def hyplog(x,theta):
    h=np.dot(x,theta)
    return h
def sigmoid(z):
    g = 1/(1 + np.exp(-1*z));
    return g
def gradlog(x,y,theta,alpha):
    his=[]
    m=len(y)
    for i in range(1000):
        
        k=sigmoid(hyplog(x,theta))-y
        g=np.dot(x.transpose(),k)/m
        theta=theta - alpha*g
        his.append(cost4(x,y,theta))
        if cost4(x,y,theta)<0.05:
            break
    return theta, his
def cost4(x,y,theta):
    m=len(y)
    k=-y*(np.log(sigmoid(hyplog(x,theta))))-(1-y)*(np.log(1-sigmoid(hyplog(x,theta))))
    j=np.sum(k)/m
    return j
def predict(x,theta):
    t=sigmoid(hyplog(x,theta))
    pre = [x>= 0.5 for x in t]
    return pre
def algorithm_3(x_train, y_train,x_test):
    l=np.ones(len(x_train))
    l=np.expand_dims(l,1)
    x_train=np.concatenate((l,x_train),axis=1)
    theta=np.zeros((5,1))
    theta,his=gradlog(x_train,y_train,theta,0.01)
    pre=predict(x_test,theta)
    pre = [x/1 for x in pre]
    return pre
#-------------------------------------
# Implement the function of the fourth algorithm below:
def algorithm_4(x_train, y_train,x_test):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    prediction=clf.predict(x_test)
    return prediction
#-------------------------------------
# Execute functions and extract outputs
prediction_1 = algorithm_1(x_train, y_train,x_test)
prediction_2 = algorithm_2(x_train, y_train,x_test)
prediction_3 = algorithm_3(x_train, y_train,x_test)
prediction_4 = algorithm_4(x_train, y_train,x_test)

# Build a new dataframe from the output of models
dataset = pd.DataFrame(data=prediction_1 , columns= ['algorithm_1'])
dataset['algorithm_2'] = prediction_2
dataset['algorithm_3'] = prediction_3
dataset['algorithm_4'] = prediction_4
dataset['Y'] = y_test.values
#-------------------------------------
# Separation of train and test date
y_train, x_train, y_test, x_test = test_train(dataset)

#-------------------------------------
#Implement the artificial neural network function below and display the results:
def ANN(x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layers.Dense(4, input_dim=4, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))
    model.fit(x_train, y_train, epochs=30, batch_size=32)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['acc'])
    return model.predict(x_test)


stacked=ANN(x_train, y_train, x_test, y_test)
print(acc(stacked,y_test))

