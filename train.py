from PIL import Image
import os
from feature import NPDFeature
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
from ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

#转化灰度图
def convertHuidu():
    i=0;
    while i<500:
        if(i<10):
            img=Image.open("/home/kodgv/第三次实验/ML2017-lab-03/datasets/original/nonface/nonface_00"+str(i)+".jpg")
        if(i>=10 and i<100):
            img = Image.open("/home/kodgv/第三次实验/ML2017-lab-03/datasets/original/nonface/nonface_0" + str(i) + ".jpg")
        if(i>=100):
            img = Image.open("/home/kodgv/第三次实验/ML2017-lab-03/datasets/original/nonface/nonface_" + str(i) + ".jpg")
        img =img.resize((24,24))
        iml=img.convert('L')
        iml.save("/home/kodgv/第三次实验/ML2017-lab-03/huidutu/non"+str(i)+".jpg")
        i+=1
#提取face特征
def exact_face():
    #处理face
    i=0;
    feature=[]
    label=[]
    while i<500:
        img_wait_deal=np.array(Image.open("/home/kodgv/第三次实验/ML2017-lab-03/huidutu/" + str(i) + ".jpg"))
        NPD=NPDFeature(img_wait_deal)
        label.append(1)
        feature.append(list(NPD.extract()))
        i+=1
    np.save("feature.npy",np.array(feature))
    np.save("label.npy",np.array(label).reshape(1,len(label)))
#提取非face特征
def exact_nonface():
    #处理nonface
    i=0;
    feature=[]
    label=[]
    while i<500:
        img_wait_deal=np.array(Image.open("/home/kodgv/第三次实验/ML2017-lab-03/huidutu/non" + str(i) + ".jpg"))
        NPD=NPDFeature(img_wait_deal)
        label.append(-1)
        feature.append(list(NPD.extract()))
        i+=1
    np.save("nonfeature.npy",np.array(feature))
    np.save("nonlabel.npy",np.array(label).reshape(1,len(label)))


if __name__ == "__main__":
    # write your code here
    #预处理，由于已经处理过，可以直接读取文件
    # convertHuidu()
    # exact_face()
    # exact_nonface()
#label {-1,+1}
    feature=np.load("feature.npy")
    label=np.load("label.npy")
    feature_non=np.load("nonfeature.npy")
    label_non=np.load("nonlabel.npy")
    label_non=label_non.reshape((label_non.shape[1]),1)
    label=label.reshape((label.shape[1],1))
    # X_train_face, X_test_face, y_train_face, y_test_face = train_test_split(feature, label, test_size=0.2, random_state=42)
    # X_train_nonface, X_test_nonface, y_train_nonface, y_test_nonface = train_test_split(feature_non, label_non, test_size=0.2, random_state=42)

    # X_train=np.vstack((X_train_face,X_train_nonface))
    # y_train=np.vstack((y_train_face,y_train_nonface))
    # X_test = np.vstack((X_test_face, X_test_nonface))
    # y_test = np.vstack((y_test_face, y_test_nonface))
    #
    # indices = np.random.permutation(X_train.shape[0])
    # X_train=X_train[indices]
    # y_train=y_train[indices]
    # indices = np.random.permutation(X_test.shape[0])
    # X_test = X_test[indices]
    # y_test = y_test[indices]

    X=np.vstack((feature,feature_non))
    y=np.vstack((label,label_non))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ada=AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3),10)
    ada.fit(X_train,y_train)
    y_test_predict=ada.predict(X_test)



    print(y_test.shape)
    print(np.sum(np.abs(y_test_predict-y_test)/2))

    with open('report.txt', "wb") as f:
        repo = classification_report(y_test, y_test_predict, target_names=["face", "nonface"])
        f.write(repo.encode())








    # mode.fit(X_train,y_train,sample_weight=w)
    # train_predict=mode.predict(X_train)
    # train_predict=np.array(train_predict).reshape(len(train_predict),1)
    # y_train_error=np.abs(train_predict-y_train)
    # print(np.dot(np.transpose(w_array), y_train_error))
    #
    # I_error=np.dot(np.transpose(w_array), y_train_error)
    # if(I_error<0.001):
    #     I_error=0.001
    # print(0.5*np.log((1-I_error)/I_error))
    # function_param.append(0.5*np.log((1-I_error)/I_error))

