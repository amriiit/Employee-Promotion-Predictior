import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([
    [1,55,2],
    [2,58,2],
    [2,60,3],
    [3,62,3],
    [4,65,4],
    [5,68,5],
    [6,72,6],
    [7,78,7],
    [8,85,8],
    [10,92,10]
])

y_train = np.array([
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1
])

#function for normalization
def normalization(x_train):
    mean=np.mean(x_train,axis=0)
    sd=np.std(x_train,axis=0)
    X_train=(x_train-mean)/sd
    return X_train,mean,sd

#function to compute sigmoid
def sigmoid(z):
    sig=1/(1+np.exp(-z))
    return sig

#fuction to compute cost
def compute_cost(x_train,y_train,w,b):
    m=x_train.shape[0]
    total_cost=0
    for i in range(m):
        z=np.dot(x_train[i],w)+b
        prediction=sigmoid(z)
        total_cost+=y_train[i]*np.log(prediction)+(1-y_train[i])*(np.log(1-prediction))
    total_cost=-(total_cost/m)
    return total_cost    

def compute_gradient(x_train,y_train,w,b):
    m,n=x_train.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        z=np.dot(x_train[i],w)+b
        prediction=sigmoid(z)
        error=prediction-y_train[i]
        for j in range(n):
            dj_dw[j]+=error*x_train[i,j]
        dj_db+=error

    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw, dj_db

def gradient_descent(x_train,y_train,w,b,iterations,alpha):
    cost_history=[]
    for i in range(iterations):
        cost=compute_cost(x_train,y_train,w,b)
        if(i%1000==0):
            print("Cost in iteration",i,":",cost)
        cost_history.append(cost)
        dj_dw,dj_db=compute_gradient(x_train,y_train,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
    return w,b,cost_history

def predict(w_final,b_final,x_prediction):
    z=np.dot(w_final,x_prediction)+b_final
    probability=sigmoid(z)
    percentage=probability*100
    return (probability>=0.5).astype(int),percentage

#training the model
x_norm,mean,sd=normalization(x_train)
w=np.zeros(x_train.shape[1])
b=0
alpha=0.01
iterations=10000
w_final,b_final,cost_history=gradient_descent(x_norm, y_train,w,b,iterations,alpha)

#taking input and making prediction through the model
X_train=[]
for i in range(x_train.shape[1]):
    a=float(input(f"Enter parameter {i+1}: "))
    X_train.append(a)
X_train=np.array(X_train)    
X_train_norm=(X_train-mean)/sd
prediction,percentage=predict(w_final,b_final,X_train_norm)
print("Prediction:",prediction)
print("Probability:",percentage,"%")

#cost vs iterations
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.grid(True)
plt.savefig("cost_vs_iterations.png")

#sigmoid curve
x=np.linspace(-10,10,100)
y=sigmoid(x)
plt.figure()
plt.plot(x,y)
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.grid(True)
plt.savefig("sigmoid_curve.png")

plt.show()