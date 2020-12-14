from MultiLayerPerceptron import *
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np
training, validation, test = mnist_loader.load_data()
offset = 5000
t = training[0][0:offset],training[1][0:offset]
offsetv = 500
v = validation[0][0:offsetv],validation[1][0:offsetv]

reps = 5

hidden_layer_size = 100
weight_range = (-0.4,0.4)
epochs = 100
learn_step = 0.05
mini_batch_size = 64
activation_function = sigmoid_function
derivative_function = sigmoid_derivative
patience = 40
momentum = 0.9


x_list = []
y_list = []

opti_names = ['Momentum','Nesterov','Adam','Adagrad','Adadelta']   
    
 

for rep in range(0,reps):

   
    


    
    m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
    _,avg_x,avg_y,_ = m.train_momentum(t,v,epochs,learn_step,64,activation_function,derivative_function,patience,momentum)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
    m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
    _,avg_x,avg_y,_ = m.train_nesterov(t,v,epochs,learn_step,64,activation_function,derivative_function,patience,momentum)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
   
    m = None
    m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
    _,avg_x,avg_y,_ = \
                    m.train_adam(t,v,epochs,0.05,
                            mini_batch_size,activation_function,derivative_function,
                            patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
    
    m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
    _,avg_x,avg_y,_ = m.train_adagrad(t,v,epochs,0.01,1,activation_function,derivative_function,patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
    m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range)
    _,avg_x,avg_y,_ = m.train_adadelta(t,v,epochs,0.01,1,activation_function,derivative_function,patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
    


 
plt.title('Skuteczność sieci w kolejnych epokach dla wszystkich optymalizatorów')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
for i in range(0,2):
    
    plt.plot(x_list[i],y_list[i],label=opti_names[i])
plt.legend()
plt.show()
plt.savefig('relu_learn_summary_500momentum.png')
plt.cla()


#WEIGHT TEST

we_names = ["Losowa", "Xavier","He"]
activation_function = sigmoid_function
derivative_function = sigmoid_derivative
x_list = []
y_list = []

for rep in range(0,reps):

   
    
    m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range,"None")
    _,avg_x,avg_y,_ = m.train(t,v,epochs,learn_step,mini_batch_size,activation_function,derivative_function,patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
    m = MultiLayerPerceptron([784,hidden_layer_size,10],None,'Xavier')
    _,avg_x,avg_y,_ = m.train(t,v,epochs,learn_step,mini_batch_size,activation_function,derivative_function,patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
    m = None
    m = MultiLayerPerceptron([784,hidden_layer_size,10],None,'He')
    _,avg_x,avg_y,_ = \
                    m.train(t,v,epochs,learn_step,
                            mini_batch_size,activation_function,derivative_function,
                            patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
plt.title('Skuteczność sieci w kolejnych epokach dla wszystkich inicjalizacji wag')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
print(x_list)
print(y_list)
print(we_names)
for i in range(0,3):
    
    plt.plot(x_list[i],y_list[i],label=we_names[i])
plt.legend()
plt.savefig('sigm_weight_summary_plus_001_t2.png')
plt.cla()



activation_function = softplus_function
derivative_function = sigmoid_function

x_list = []
y_list = []

for rep in range(0,reps):

   
  
    m = MultiLayerPerceptron([784,hidden_layer_size,10],weight_range,"None")
    _,avg_x,avg_y,_ = m.train(t,v,epochs,learn_step,mini_batch_size,activation_function,derivative_function,patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
    m = MultiLayerPerceptron([784,hidden_layer_size,10],None,'Xavier')
    _,avg_x,avg_y,_ = m.train(t,v,epochs,learn_step,mini_batch_size,activation_function,derivative_function,patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
   
    m = None
    m = MultiLayerPerceptron([784,hidden_layer_size,10],None,'He')
    _,avg_x,avg_y,_ = \
                    m.train(t,v,epochs,learn_step,
                            mini_batch_size,activation_function,derivative_function,
                            patience)
    avg_x = [element / reps for element in x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in y]
    y_list.append(avg_y)
    
plt.title('Skuteczność sieci w kolejnych epokach dla wszystkich inicjalizacji wag')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
print(x_list)
print(y_list)
print(we_names)
for i in range(0,3):
    
    plt.plot(x_list[i],y_list[i],label=we_names[i])
plt.legend()
plt.savefig('relu_weight_summary_plus_001_t2.png')
plt.cla()
