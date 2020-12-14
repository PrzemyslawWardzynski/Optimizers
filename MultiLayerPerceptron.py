import random


import numpy as np
import time
import math

class MultiLayerPerceptron(object):

    def __init__(self, layerSizes,weight_range,weight_initialization='None'):
        self.layerSizes = layerSizes
        self.weights = []
        self.biases = []
        for i in range(0,len(layerSizes)-1):

            if weight_range == None:
                #Xavier
                weight_variance = 0.1
                if weight_initialization == 'Xavier':
                    
                    weight_variance += 2 / (layerSizes[i]+ layerSizes[i+1])
              
                #He
                elif weight_initialization == 'He':
                    

                    weight_variance += 2 / layerSizes[i]            
                
               
                self.weights.append(math.sqrt(weight_variance) * np.random.randn(layerSizes[i+1],layerSizes[i]))
                weight_variance = 0
                self.biases.append(math.sqrt(weight_variance) * np.random.randn(layerSizes[i+1]))
            else:
                low,high = weight_range
                
                self.weights.append(np.random.uniform(low,high,(layerSizes[i+1],layerSizes[i])))
                weight_variance = 0
                self.biases.append(np.random.uniform(low,high,layerSizes[i+1]))
        
    def feedforward(self, input,  activation_function):

        z_list = []
        activations_list = [input]

        wb_size = len(self.weights)
        for i in range (0, wb_size-1):
            z = np.dot(self.weights[i], input)+self.biases[i]
            z_list.append(z)
            input = activation_function(z)
            activations_list.append(input)
            
        #calculate outer layer with softmax func
        z = np.dot(self.weights[wb_size-1], input)+self.biases[wb_size-1]
        z_list.append(z)
        input = softmax(z)
        activations_list.append(input)
        return input,z_list,activations_list

    def backpropagation(self, x, y, activation_function, derivative_function):

        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # forward propagation, save activations and z

        net_y,z_list,activations_list = self.feedforward(x,activation_function)
        
        # outer layer error - activation function - softmax
        y_hotone = np.zeros(10)
        y_hotone[y] = 1
        delta = net_y - y_hotone
        
        
        gradient_biases[-1] = delta
        gradient_weights[-1] = np.outer(delta ,(activations_list[-2]).T)
    
        # error propagation

        for i in range (len(self.weights)-2,-1):
            z = z_list[i]
            act_deriv = derivative_function(z)
            delta = self.weights[i+1].T @ delta * act_deriv
            gradient_biases[i] = delta
            gradient_weights[i] = np.outer(delta, (activations_list[i]).T)

        return gradient_biases,gradient_weights

    def train(self, training_data, validation_data, epochs, learn_step, minibatch_size, activation_function, derivative_function, patience):

        train_data_length = len(training_data[0])
        if validation_data:
            validation_data_length = len(validation_data[0])
        max_accuracy = 0.0
        max_epoch = epochs

        reversed_accuracy_list = []
        reversed_epoch_list = []

        for i in range(0,epochs):

            
            minibatches = [
                (training_data[0][j:j+minibatch_size],
                training_data[1][j:j+minibatch_size])
                for j in range(0, train_data_length, minibatch_size)]
            
            
            for minibatch in minibatches:
                
                gradient_b = [np.zeros(b.shape) for b in self.biases]
                gradient_w = [np.zeros(w.shape) for w in self.weights]
                
               # start = time.time()
                minibatch_len = len(minibatch[0])
                for l in range(minibatch_len):
                    
                    x = minibatch[0][l]
                    y = minibatch[1][l]
                    #startT = time.time_ns()
                    delta_gradient_b, delta_gradient_w = self.backpropagation(x,y,activation_function,derivative_function)
                   # end = time.time_ns() - startT
                    gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
                    gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

                #print("BATCH TIME: {0}".format(time.time() - start))
                #print("{2},{3},PROP TO BATCH : {0} | {1}".format(end*minibatch_len/10**9,(time.time()-start),i,l))
               # start = time.time()
                self.weights = [w-(learn_step/minibatch_len)*nw
                        for w, nw in zip(self.weights, gradient_w)]
                self.biases = [b-(learn_step/minibatch_len)*nb
                       for b, nb in zip(self.biases, gradient_b)]
                
            
                #print("UPDATE TIME: {0}".format(time.time() - start))         
            if validation_data:
                #print(self.weights)
                #print(self.biases)
                #start = time.time()
                accuracy = self.accuracy(validation_data,validation_data_length,activation_function)
                #print("VALIDATION TIME: {0}".format(time.time() - start))
                accuracy = round(accuracy,2)
                reversed_accuracy_list.insert(0,accuracy)
                reversed_epoch_list.insert(0,i)
                print ("Epoch {0}: {1}".format(
                    i, accuracy))
            else:
                print ("Epoch {0} complete".format(i))

                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
            if (accuracy > max_accuracy):
                #print("----------------{0}".format(i))
                max_epoch = i
                max_accuracy = accuracy
                
            if(i - max_epoch > patience):
                print("PATIENCE")
                
                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
                
       
        return epochs,reversed_epoch_list,reversed_accuracy_list,max_accuracy 
#Momentum
    def train_momentum(self, training_data, validation_data, epochs, learn_step, minibatch_size, activation_function, derivative_function, patience, momentum):

        train_data_length = len(training_data[0])
        if validation_data:
            validation_data_length = len(validation_data[0])
        max_accuracy = 0.0
        max_epoch = epochs

        reversed_accuracy_list = []
        reversed_epoch_list = []

        prev_gradient_b = [np.zeros(b.shape) for b in self.biases]
        prev_gradient_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(0,epochs):
            #print(self.biases)
            
            minibatches = [
                (training_data[0][j:j+minibatch_size],
                training_data[1][j:j+minibatch_size])
                for j in range(0, train_data_length, minibatch_size)]
            
            

            for minibatch in minibatches:
                
                gradient_b = [np.zeros(b.shape) for b in self.biases]
                gradient_w = [np.zeros(w.shape) for w in self.weights]
                
               # start = time.time()
                minibatch_len = len(minibatch[0])
                for l in range(minibatch_len):
                    
                    x = minibatch[0][l]
                    y = minibatch[1][l]
                    #startT = time.time_ns()
                    delta_gradient_b, delta_gradient_w = self.backpropagation(x,y,activation_function,derivative_function)
                   # end = time.time_ns() - startT
                    gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
                    gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

                #print("BATCH TIME: {0}".format(time.time() - start))
                #print("{2},{3},PROP TO BATCH : {0} | {1}".format(end*minibatch_len/10**9,(time.time()-start),i,l))
               # start = time.time()
                self.weights = [w-(learn_step/minibatch_len)*nw
                        for w, nw in zip(self.weights, gradient_w)]
                self.biases = [b-(learn_step/minibatch_len)*nb
                       for b, nb in zip(self.biases, gradient_b)]
                self.weights = [w-(momentum/minibatch_len)*nw
                        for w, nw in zip(self.weights, prev_gradient_w)]
                self.biases = [b-(momentum/minibatch_len)*nb
                       for b, nb in zip(self.biases, prev_gradient_b)]
                prev_gradient_b = gradient_b    
                prev_gradient_w = gradient_w

                #print("UPDATE TIME: {0}".format(time.time() - start))         
            if validation_data:
                #print(self.weights)
                #print(self.biases)
                #start = time.time()
                accuracy = self.accuracy(validation_data,validation_data_length,activation_function)
                #print("VALIDATION TIME: {0}".format(time.time() - start))
                accuracy = round(accuracy,2)
                reversed_accuracy_list.insert(0,accuracy)
                reversed_epoch_list.insert(0,i)
                print ("Epoch {0}: {1}".format(
                    i, accuracy))
            else:
                print ("Epoch {0} complete".format(i))

                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
            if (accuracy > max_accuracy):
                #print("----------------{0}".format(i))
                max_epoch = i
                max_accuracy = accuracy
                
            if(i - max_epoch > patience):
                print("PATIENCE")
                
                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
                
       
        return epochs,reversed_epoch_list,reversed_accuracy_list,max_accuracy 

#Nesterov
    def train_nesterov(self, training_data, validation_data, epochs, learn_step, minibatch_size, activation_function, derivative_function, patience, momentum):

        train_data_length = len(training_data[0])
        if validation_data:
            validation_data_length = len(validation_data[0])
        max_accuracy = 0.0
        max_epoch = epochs

        reversed_accuracy_list = []
        reversed_epoch_list = []

        prev_gradient_b = [np.zeros(b.shape) for b in self.biases]
        prev_gradient_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(0,epochs):

            
            minibatches = [
                (training_data[0][j:j+minibatch_size],
                training_data[1][j:j+minibatch_size])
                for j in range(0, train_data_length, minibatch_size)]
            
            

            for minibatch in minibatches:
                
                gradient_b = [np.zeros(b.shape) for b in self.biases]
                gradient_w = [np.zeros(w.shape) for w in self.weights]
                
               # start = time.time()
                minibatch_len = len(minibatch[0])
                for l in range(minibatch_len):
                    
                    x = minibatch[0][l]
                    y = minibatch[1][l]
                    #startT = time.time_ns()
                    delta_gradient_b, delta_gradient_w = self.backpropagation(x,y,activation_function,derivative_function)
                   # end = time.time_ns() - startT
                    gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
                    gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

                #print("BATCH TIME: {0}".format(time.time() - start))
                #print("{2},{3},PROP TO BATCH : {0} | {1}".format(end*minibatch_len/10**9,(time.time()-start),i,l))
               # start = time.time()
                self.weights = [w-(learn_step*(1+momentum)/minibatch_len)*nw
                        for w, nw in zip(self.weights, gradient_w)]
                self.biases = [b-(learn_step*(1+momentum)/minibatch_len)*nb
                       for b, nb in zip(self.biases, gradient_b)]
                self.weights = [w-(momentum*momentum/minibatch_len)*nw
                        for w, nw in zip(self.weights, prev_gradient_w)]
                self.biases = [b-(momentum*momentum/minibatch_len)*nb
                       for b, nb in zip(self.biases, prev_gradient_b)]
                prev_gradient_b = gradient_b    
                prev_gradient_w = gradient_w

                #print("UPDATE TIME: {0}".format(time.time() - start))         
            if validation_data:
                #print(self.weights)
                #print(self.biases)
                #start = time.time()
                accuracy = self.accuracy(validation_data,validation_data_length,activation_function)
                #print("VALIDATION TIME: {0}".format(time.time() - start))
                accuracy = round(accuracy,2)
                reversed_accuracy_list.insert(0,accuracy)
                reversed_epoch_list.insert(0,i)
                print ("Epoch {0}: {1}".format(
                    i, accuracy))
            else:
                print ("Epoch {0} complete".format(i))

                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
            if (accuracy > max_accuracy):
                #print("----------------{0}".format(i))
                max_epoch = i
                max_accuracy = accuracy
                
            if(i - max_epoch > patience):
                print("PATIENCE")
                
                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
                
       
        return epochs,reversed_epoch_list,reversed_accuracy_list,max_accuracy 
#Adagrad
    def train_adagrad(self, training_data, validation_data, epochs, learn_step, minibatch_size, activation_function, derivative_function, patience):

        train_data_length = len(training_data[0])
        if validation_data:
            validation_data_length = len(validation_data[0])
        max_accuracy = 0.0
        max_epoch = epochs

        reversed_accuracy_list = []
        reversed_epoch_list = []

        eps = 1e-8
        gt_b = [np.zeros(b.shape) for b in self.biases]
        gt_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(0,epochs):

            
            minibatches = [
                (training_data[0][j:j+minibatch_size],
                training_data[1][j:j+minibatch_size])
                for j in range(0, train_data_length, minibatch_size)]
            
            
            for minibatch in minibatches:
                
                gradient_b = [np.zeros(b.shape) for b in self.biases]
                gradient_w = [np.zeros(w.shape) for w in self.weights]
                
               # start = time.time()
                minibatch_len = len(minibatch[0])
                for l in range(minibatch_len):
                    
                    x = minibatch[0][l]
                    y = minibatch[1][l]
                    #startT = time.time_ns()
                    delta_gradient_b, delta_gradient_w = self.backpropagation(x,y,activation_function,derivative_function)
                   # end = time.time_ns() - startT
                    gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
                    gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

                #print("BATCH TIME: {0}".format(time.time() - start))
                #print("{2},{3},PROP TO BATCH : {0} | {1}".format(end*minibatch_len/10**9,(time.time()-start),i,l))
               # start = time.time()

                gt_b = [nb+dnb*dnb for nb, dnb in zip(gt_b, gradient_b)]
                gt_w = [nw+dnw*dnw for nw, dnw in zip(gt_w, gradient_w)]

                self.weights = [w-(learn_step/(np.sqrt(gt_dw + eps)*minibatch_len))*nw
                        for w, nw,gt_dw in zip(self.weights, gradient_w, gt_w)]
                self.biases = [b-(learn_step/(np.sqrt(gt_db + eps)*minibatch_len))*nb
                       for b, nb,gt_db in zip(self.biases, gradient_b, gt_b)]

            print
                
                
            
                #print("UPDATE TIME: {0}".format(time.time() - start))         
            if validation_data:
                #print(self.weights)
                #print(self.biases)
                #start = time.time()
                accuracy = self.accuracy(validation_data,validation_data_length,activation_function)
                #print("VALIDATION TIME: {0}".format(time.time() - start))
                accuracy = round(accuracy,2)
                reversed_accuracy_list.insert(0,accuracy)
                reversed_epoch_list.insert(0,i)
                print ("Epoch {0}: {1}".format(
                    i, accuracy))
            else:
                print ("Epoch {0} complete".format(i))

                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
            if (accuracy > max_accuracy):
                #print("----------------{0}".format(i))
                max_epoch = i
                max_accuracy = accuracy
                
            if(i - max_epoch > patience):
                print("PATIENCE")
                
                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
                
       
        return epochs,reversed_epoch_list,reversed_accuracy_list,max_accuracy 
#Adaline
    def train_adadelta(self, training_data, validation_data, epochs, learn_step, minibatch_size, activation_function, derivative_function, patience):

        train_data_length = len(training_data[0])
        if validation_data:
            validation_data_length = len(validation_data[0])
        max_accuracy = 0.0
        max_epoch = epochs

        reversed_accuracy_list = []
        reversed_epoch_list = []

        eps = 1e-8
        delta = 0.9
        g_avg_b = [np.zeros(b.shape) for b in self.biases]
        g_avg_w = [np.zeros(w.shape) for w in self.weights]   
        u_avg_b = [np.zeros(b.shape) for b in self.biases]
        u_avg_w = [np.zeros(w.shape) for w in self.weights]


        for i in range(0,epochs):

            
            minibatches = [
                (training_data[0][j:j+minibatch_size],
                training_data[1][j:j+minibatch_size])
                for j in range(0, train_data_length, minibatch_size)]
            
            
            

            for minibatch in minibatches:
                
                gradient_b = [np.zeros(b.shape) for b in self.biases]
                gradient_w = [np.zeros(w.shape) for w in self.weights]
                
               # start = time.time()
                minibatch_len = len(minibatch[0])
                for l in range(minibatch_len):
                    
                    x = minibatch[0][l]
                    y = minibatch[1][l]
                    #startT = time.time_ns()
                    delta_gradient_b, delta_gradient_w = self.backpropagation(x,y,activation_function,derivative_function)
                   # end = time.time_ns() - startT
                    gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
                    gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

                #print("BATCH TIME: {0}".format(time.time() - start))
                #print("{2},{3},PROP TO BATCH : {0} | {1}".format(end*minibatch_len/10**9,(time.time()-start),i,l))
               # start = time.time()
                
                
                g_avg_b = [delta * avg + (1.-delta)*gb*gb for avg,gb in zip(g_avg_b,gradient_b)]
                g_avg_w = [delta * avg + (1.-delta)*gw*gw for avg,gw in zip(g_avg_w,gradient_w)]
                update_b = [gb * ((np.sqrt(u_avg + eps)) / np.sqrt(avg + eps)) for gb, u_avg, avg in zip(gradient_b, u_avg_b,g_avg_b)]
                update_w = [gw * ((np.sqrt(u_avg + eps)) / np.sqrt(avg + eps)) for gw, u_avg, avg in zip(gradient_w, u_avg_w,g_avg_w)]
                
                u_avg_b = [delta * avg + (1.-delta)*gb*gb for avg,gb in zip(u_avg_b,update_b)]
                u_avg_w = [delta * avg + (1.-delta)*gw*gw for avg,gw in zip(u_avg_w,update_w)]


                self.weights = [w-(uw/minibatch_len)
                        for w, uw in zip(self.weights, update_w)]
                self.biases = [b-(ub/minibatch_len)
                       for b, ub in zip(self.biases, update_b)]
                
            #print(self.biases)  
                    

                #print("UPDATE TIME: {0}".format(time.time() - start))         
            if validation_data:
                #print(self.weights)
                #print(self.biases)
                #start = time.time()
                accuracy = self.accuracy(validation_data,validation_data_length,activation_function)
                #print("VALIDATION TIME: {0}".format(time.time() - start))
                accuracy = round(accuracy,2)
                reversed_accuracy_list.insert(0,accuracy)
                reversed_epoch_list.insert(0,i)
                print ("Epoch {0}: {1}".format(
                    i, accuracy))
            else:
                print ("Epoch {0} complete".format(i))

                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
            if (accuracy > max_accuracy):
                #print("----------------{0}".format(i))
                max_epoch = i
                max_accuracy = accuracy
                
            if(i - max_epoch > patience):
                print("PATIENCE")
                
                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
                
       
        return epochs,reversed_epoch_list,reversed_accuracy_list,max_accuracy 
#Adam
    def train_adam(self, training_data, validation_data, epochs, learn_step, minibatch_size, activation_function, derivative_function, patience):

        train_data_length = len(training_data[0])
        if validation_data:
            validation_data_length = len(validation_data[0])
        max_accuracy = 0.0
        max_epoch = epochs

        reversed_accuracy_list = []
        reversed_epoch_list = []

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        timestep = 1

        m_b = [np.zeros(b.shape) for b in self.biases]
        m_w = [np.zeros(w.shape) for w in self.weights]   
        v_b = [np.zeros(b.shape) for b in self.biases]
        v_w = [np.zeros(w.shape) for w in self.weights]
        adjusted_m_b = [np.zeros(b.shape) for b in self.biases]
        adjusted_m_w = [np.zeros(w.shape) for w in self.weights]
        adjusted_v_b = [np.zeros(b.shape) for b in self.biases]
        adjusted_v_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(0,epochs):

            
            minibatches = [
                (training_data[0][j:j+minibatch_size],
                training_data[1][j:j+minibatch_size])
                for j in range(0, train_data_length, minibatch_size)]
            
            
            for minibatch in minibatches:
                
                gradient_b = [np.zeros(b.shape) for b in self.biases]
                gradient_w = [np.zeros(w.shape) for w in self.weights]
                
               # start = time.time()
                minibatch_len = len(minibatch[0])
                for l in range(minibatch_len):
                    
                    x = minibatch[0][l]
                    y = minibatch[1][l]
                    #startT = time.time_ns()
                    delta_gradient_b, delta_gradient_w = self.backpropagation(x,y,activation_function,derivative_function)
                   # end = time.time_ns() - startT
                    gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
                    gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

                #print("BATCH TIME: {0}".format(time.time() - start))
                #print("{2},{3},PROP TO BATCH : {0} | {1}".format(end*minibatch_len/10**9,(time.time()-start),i,l))
               # start = time.time()
                m_b = [beta1 * mb + (1.-beta1)*gb for mb,gb in zip(m_b,gradient_b)]
                v_b = [beta2 * vb + (1.-beta2)*gb*gb for vb,gb in zip(v_b,gradient_b)]

                m_w = [beta1 * mw + (1.-beta1)*gw for mw,gw in zip(m_w,gradient_w)]
                v_w = [beta2 * vw + (1.-beta2)*gw*gw for vw,gw in zip(v_w,gradient_w)]


                adjusted_m_b = [mb / (1. - beta1**timestep) for mb in m_b]
                adjusted_v_b = [vb / (1. - beta2**timestep) for vb in v_b]

                adjusted_m_w = [mw / (1. - beta1**timestep) for mw in m_w]
                adjusted_v_w = [vw / (1. - beta2**timestep) for vw in v_w]
                
                

                

                self.weights = [w-(learn_step*adm/(minibatch_len*(np.sqrt(adv)+eps)))
                       for w, adm, adv in zip(self.weights, adjusted_m_w,adjusted_v_w)]
                self.biases = [b-(learn_step*adm/(minibatch_len*(np.sqrt(adv)+eps)))
                       for b, adm, adv in zip(self.biases, adjusted_m_b,adjusted_v_b)]
                timestep += 1       
                
            

                #print("UPDATE TIME: {0}".format(time.time() - start))         
            if validation_data:
                #print(self.weights)
                #print(self.biases)
                #start = time.time()
                accuracy = self.accuracy(validation_data,validation_data_length,activation_function)
                #print("VALIDATION TIME: {0}".format(time.time() - start))
                accuracy = round(accuracy,2)
                reversed_accuracy_list.insert(0,accuracy)
                reversed_epoch_list.insert(0,i)
                print ("Epoch {0}: {1}".format(
                    i, accuracy))
            else:
                print ("Epoch {0} complete".format(i))

                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
            if (accuracy > max_accuracy):
                #print("----------------{0}".format(i))
                max_epoch = i
                max_accuracy = accuracy
                
            if(i - max_epoch > patience):
                print("PATIENCE")
                
                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy

        return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy

    def accuracy(self, validation_data, data_size, activation_function):

        hit_counter = 0

        for i in  range(data_size):
            x = validation_data[0][i]
            y = validation_data[1][i]
            net_y,_,_ = self.feedforward(x,activation_function)
            
            if np.argmax(net_y) == y:
                hit_counter += 1
        return float(hit_counter) / data_size

    



    def save(self):
        np.save('biases',self.biases)
        np.save('weights',self.weights)

    def load(self):
        self.biases = ('biases.npy')
        self.weights = ('weights.npy')

    
        





    
def softmax(z):
    shift = z - np.max(z)
    exp_z = np.exp(shift)
    return exp_z/np.sum(exp_z)
    
def sigmoid_function(z):
    
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    
    return sigmoid_function(z)*(1-sigmoid_function(z))

def tanh_function(z):
    
    return 2.0/(1.0+np.exp(-2*z)) - 1

def ReLU_function(z):

    return np.where(z < 0, 0, z)

def softplus_function(z):

    return np.log(1 + np.exp(z))



