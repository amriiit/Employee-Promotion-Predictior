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

#fuction to compute cost(non-regularized)
def compute_cost(x_train,y_train,w,b):
    m=x_train.shape[0]
    total_cost=0
    for i in range(m):
        z=np.dot(x_train[i],w)+b
        prediction=sigmoid(z)
        total_cost+=y_train[i]*np.log(prediction)+(1-y_train[i])*(np.log(1-prediction))
    total_cost=-(total_cost/m)
    return total_cost  

#fuction to compute cost(regularized)
def compute_cost_regularized(x_train,y_train,w,b,lambdaa):
    m=x_train.shape[0]
    total_cost=0
    for i in range(m):
        z=np.dot(x_train[i],w)+b
        prediction=sigmoid(z)
        total_cost+=y_train[i]*np.log(prediction)+(1-y_train[i])*(np.log(1-prediction))
    total_cost=-(total_cost/m)
    regularization=0
    for j in range(len(w)):
        regularization+=w[j]**2
    regularization=(lambdaa/(2*m))*regularization
    return total_cost+regularization   

#fuction to compute gradient
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
    
#fuction to compute gradient(regularized) 
def compute_gradient_regularized(x_train,y_train,w,b,lambdaa):
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

    for j in range(n):
        dj_dw[j]=(dj_dw[j]/m+(lambdaa/m)*w[j])
    dj_db=dj_db/m
    return dj_dw, dj_db   

#function to compute gradient descent
def gradient_descent(x_train,y_train,w,b,iterations,alpha):
    cost_history=[]
    print("Cost per 1000th iteration for non-regularized model:")
    for i in range(iterations):
        cost=compute_cost(x_train,y_train,w,b)
        if(i%1000==0):
            print("Cost in iteration",i,":",cost)
        cost_history.append(cost)
        dj_dw,dj_db=compute_gradient(x_train,y_train,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
    return w,b,cost_history

#function to compute gradient descent(regularized)
def gradient_descent_regularized(x_train,y_train,w,b,iterations,alpha,lambdaa):
    cost_history_regularized=[]
    print("Cost per 1000th iteration for regularized model:")
    for i in range(iterations):
        cost=compute_cost_regularized(x_train,y_train,w,b,lambdaa)
        if(i%1000==0):
            print("Cost in iteration",i,":",cost)
        cost_history_regularized.append(cost)
        dj_dw,dj_db=compute_gradient_regularized(x_train,y_train,w,b,lambdaa)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
    return w,b,cost_history_regularized

#function for prediction
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
lambdaa=0.1
iterations=10000
w_final,b_final,cost_history=gradient_descent(x_norm, y_train,w,b,iterations,alpha)
w_final_regularized,b_final_regularized,cost_history_regularized=gradient_descent_regularized(x_norm, y_train,w,b,iterations,alpha,lambdaa)

#taking input and making prediction through the model
X_train=[]
for i in range(x_train.shape[1]):
    a=float(input(f"Enter parameter {i+1}: "))
    X_train.append(a)
X_train=np.array(X_train)    
X_train_norm=(X_train-mean)/sd
prediction,percentage=predict(w_final,b_final,X_train_norm)
prediction_regularized,percentage_regularized=predict(w_final_regularized,b_final_regularized,X_train_norm)
print("Prediction(Non-Regularized):",prediction)
print("Probability(Non-Regularized):",percentage,"%")
print("Prediction(Regularized):",prediction_regularized)
print("Probability(Regularized):",percentage_regularized,"%")


print("\nWeights (Non-Regularized):")
print(w_final)
print("\nWeights (Regularized):")
print(w_final_regularized)


print("\n===== MODEL COMPARISON =====")
print("Final Cost (Non-Regularized):",cost_history[-1])
print("Final Cost (Regularized):",cost_history_regularized[-1])

#code to compare accuracy of the two models
def accuracy(x_train,y_train,w,b):
    correct=0
    m=x_train.shape[0]
    for i in range(m):
        prediction,_=predict(w,b,x_train[i])
        if prediction==y_train[i]:
            correct+=1
    return (correct/m)*100


print("\nWeight Magnitudes")
print("Non-Regularized:",np.sum(np.abs(w_final)))
print("Regularized:",np.sum(np.abs(w_final_regularized)))

accuracy_normal=accuracy(x_norm,y_train,w_final,b_final)
accuracy_regularized=accuracy(x_norm,y_train,w_final_regularized,b_final_regularized)
print("\nAccuracy (Non-Regularized):",accuracy_normal,"%")
print("Accuracy (Regularized):",accuracy_regularized,"%")

#iterations vs cost for regularized and non-regularized
plt.figure()
plt.plot(cost_history,label="Non-Regularized")
plt.plot(cost_history_regularized,label="Regularized")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Regularization Comparison")
plt.legend()
plt.grid(True)
plt.savefig("regularization_comparison.png")

#iterations vs cost for regularized and non-regularized
plt.figure()
plt.plot(cost_history,label="Non-Regularized")
plt.plot(cost_history_regularized,label="Regularized")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Regularization Comparison")
plt.legend()
plt.grid(True)
plt.savefig("regularization_comparison.png")


plt.figure()
plt.bar(["Exp","Perf","Projects"],np.abs(w_final),label="Non-Regularized",alpha=0.7)
plt.bar(["Exp","Perf","Projects"],np.abs(w_final_regularized),label="Regularized",alpha=0.7)
plt.title("Weight Comparison")
plt.legend()
plt.savefig("weight_comparison.png")

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