import torch
from torch.autograd.functional import jacobian
import numpy as np
import pandas as pd
import datetime
import torch.nn as nn
import matplotlib.pyplot as plt

class PIELM:

    def __init__(self,n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh",controls=False,physics=False):
        # if len(functions)==output_size:
        #     raise ValueError("gotta match number of states predicted and diferential equations")
        # self.functions = functions
        self.length= length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nodes = n_nodes
        self.W = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_w-low_w)+low_w)
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_b-low_b)+low_b)
        self.betas = torch.ones(size=(output_size*n_nodes,),requires_grad=True,dtype=torch.float)
        self.controls = controls
        self.physics = physics
        
    def train(self,accuracy, n_iterations,x_train,y_train,l,rho,steering_angle,slip_angle,speed_x,speed_y,heading_ratio,lambda_=1):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        count = 0
        error = 100
        self.lambda_ = lambda_
        
        z0 = -1
        zf = 1
        t0 = x_train[0]
        tf = x_train[-1]
        c = (zf-z0)/(tf-t0)
        x_train = z0+c*(x_train-t0)

        self.c = c
        self.x_train = torch.tensor(x_train,dtype=torch.float).reshape(x_train.shape[0],1)
        
        self.y_train = torch.tensor(y_train,dtype=torch.float)
        self.x_train_pred = self.x_train[:len(self.x_train)-self.length,]
        
        self.y_train_pred = self.y_train[:len(self.y_train)-self.length,]
        self.steering_angle = torch.tensor(steering_angle,dtype=torch.float)
        self.slip_angle = torch.tensor(slip_angle,dtype=torch.float)
        self.speed_x = torch.tensor(speed_x,dtype=torch.float)
        self.speed_y = torch.tensor(speed_y,dtype=torch.float)
        self.heading_ratio = torch.tensor(heading_ratio,dtype=torch.float)
        self.l = torch.tensor(l,dtype=torch.float)
        self.rho = torch.tensor(rho,dtype=torch.float)
        print(self.betas.is_cuda)
        print("number of samples:",len(self.x_train))
        while count < n_iterations:
            
            with torch.no_grad():
                
                jac = jacobian(self.predict_jacobian,self.betas)
                loss = self.predict_loss(self.x_train,self.y_train_pred,self.x_train_pred)
                pinv_jac = torch.linalg.pinv(jac)
                delta = torch.matmul(pinv_jac,loss)

                self.betas -=delta*0.1
            if count %10==0:
                print(loss.abs().max(dim=0),loss.mean(dim=0))
                #print(torch.mean(loss[0:4]))
                #print(torch.max(loss[0:4]))
                #print(torch.min(loss[0:4]))
                print("final loss:",(loss**2).mean())
                print(count)
            count +=1
        # print(loss[0:20],"x position")
        # print(loss[20:40],"y position")
        # print(loss[40:60],"angle")
        # print(loss[60:80],"steering")        
        # print(loss[80:100],"x speed")
        # print(loss[100:120],"y speed")
        # print(loss[120:140],"angular speed")
        # print(loss[140:160],"delta rate")        
        
        
        # for epoch in range(n_iterations):
        
            
        #     self.optimizer.zero_grad()
        #     outputs = self.forward(x_train_data)
        #     loss = self.criterion(outputs, y_train_data)
        #     loss.backward()
        #     self.optimizer.step()
            
        #     #Print training statistics
        #     if (epoch+1) % 10 == 0:
        #         print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
        #             .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))

    
    def predict_jacobian(self,betas):
        

        hx = torch.matmul(self.get_h(self.x_train_pred),betas[0:self.nodes])
        hy = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes:2*self.nodes])
        htheta = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*2:3*self.nodes])
        htheta_full = torch.matmul(self.get_h(self.x_train),betas[self.nodes*2:3*self.nodes])
        # hdelta = torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        # hdelta_full = torch.matmul(self.get_h(self.x_train),betas[self.nodes*3:4*self.nodes])
        l_pred_x = self.y_train_pred[:,0]-hx
        l_pred_y = self.y_train_pred[:,1]-hy
        l_pred_theta = self.y_train_pred[:,2]-htheta
        # l_pred_delta = self.y_train_pred[:,3]-hdelta
        #l_pred_delta = self.y_train[:,3]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        
        dhx = self.c*torch.matmul(self.get_dh(self.x_train),betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes:2*self.nodes])
        dhtheta = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*2:3*self.nodes])
        # dhdelta = self.c*torch.matmul(self.get_dh(self.x_train),betas[self.nodes*3:4*self.nodes])

        l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full+self.slip_angle)
        l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full+self.slip_angle)
        l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l


        # l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full)
        # l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full)
        # l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(hdelta_full)/self.l
        # l_delta = dhdelta-self.rho
        
        l_pred_dhx = self.speed_x- dhx
        l_pred_dhy = self.speed_y- dhy 
        l_pred_dhtheta = self.heading_ratio - dhtheta
          
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        if self.controls and self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            l_theta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_theta[0:len(self.x_train_pred)]
            l_theta[len(self.x_train_pred):]=self.lambda_*l_theta[len(self.x_train_pred):]
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                                (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,l_theta))
            
        if self.controls and not self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta))
            
        if not self.controls and not self.physics:
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta))
                                
        return loss
            
    def predict_loss(self,x,y,x_pred):
       

        hx = torch.matmul(self.get_h(x_pred),self.betas[0:self.nodes])
        hy = torch.matmul(self.get_h(x_pred),self.betas[self.nodes:2*self.nodes])
        htheta = torch.matmul(self.get_h(x_pred),self.betas[self.nodes*2:3*self.nodes])
        htheta_full = torch.matmul(self.get_h(self.x_train),self.betas[self.nodes*2:3*self.nodes])
        # hdelta = torch.matmul(self.get_h(self.x_train_pred),self.betas[self.nodes*3:4*self.nodes])
        # hdelta_full = torch.matmul(self.get_h(self.x_train),self.betas[self.nodes*3:4*self.nodes])
        l_pred_x = y[:,0]-hx
        l_pred_y = y[:,1]-hy
        l_pred_theta = y[:,2]-htheta
        # l_pred_delta = self.y_train_pred[:,3]-hdelta
        #l_pred_delta = self.y_train[:,3]-torch.matmul(self.get_h(self.x_train_pred),betas[self.nodes*3:4*self.nodes])
        
        dhx = self.c*torch.matmul(self.get_dh(x),self.betas[0:self.nodes])
        dhy = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes:2*self.nodes])
        dhtheta = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes*2:3*self.nodes])
        # dhdelta = self.c*torch.matmul(self.get_dh(x),self.betas[self.nodes*3:4*self.nodes])
    
        l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full+self.slip_angle)
        l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full+self.slip_angle)
        l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(self.steering_angle)*torch.cos(self.slip_angle)/self.l
        
        # l_x = dhx-(dhx**2+dhy**2)**(1/2)*torch.cos(htheta_full)
        # l_y = dhy-(dhx**2+dhy**2)**(1/2)*torch.sin(htheta_full)
        # l_theta = dhtheta-(dhx**2+dhy**2)**(1/2)*torch.tan(hdelta_full)/self.l
        l_pred_dhtheta = self.heading_ratio - dhtheta
        l_pred_dhx = self.speed_x- dhx
        l_pred_dhy = self.speed_y- dhy 
        # l_delta = dhdelta-self.rho
        
        #loss= torch.hstack((l_pred_x,l_pred_y,l_pred_theta,l_pred_delta,l_x,l_y,l_theta,l_delta))
        # loss= torch.hstack((l_pred_x,l_pred_y,l_x,l_y))
        # loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
        #                     self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,\
        #                     (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,(1-self.lambda_)*l_theta))
        if self.controls and self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            l_theta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_theta[0:len(self.x_train_pred)]
            l_theta[len(self.x_train_pred):]=self.lambda_*l_theta[len(self.x_train_pred):]
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta,\
                                (1-self.lambda_)*l_x,(1-self.lambda_)*l_y,l_theta))
        if self.controls and not self.physics:
            l_pred_dhtheta[0:len(self.x_train_pred)]=(1-self.lambda_)*l_pred_dhtheta[0:len(self.x_train_pred)]
            l_pred_dhtheta[len(self.x_train_pred):]=self.lambda_*l_pred_dhtheta[len(self.x_train_pred):]
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta,\
                                self.lambda_*l_pred_dhx,self.lambda_*l_pred_dhy,self.lambda_*l_pred_dhtheta))
        if not self.controls and not self.physics:
            self.lambda_ = 1
            loss= torch.hstack((self.lambda_*l_pred_x,self.lambda_*l_pred_y,self.lambda_*l_pred_theta))

        return loss    
    
    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))
    
    def pred(self,x):
        
        z0 = -1
        t0 = x[0]
        x = z0+self.c*(x-t0)

        x = torch.tensor(np.array(x),dtype=torch.float).reshape(x.shape[0],1)
        
        x_pred = torch.matmul(self.get_h(x),self.betas[0:self.nodes]) 
        y_pred = torch.matmul(self.get_h(x),self.betas[self.nodes:2*self.nodes])
        theta_pred = torch.matmul(self.get_h(x),self.betas[self.nodes*2:3*self.nodes])
        delta_pred = torch.matmul(self.get_h(x),self.betas[self.nodes*3:4*self.nodes])
        return torch.vstack((x_pred,y_pred,theta_pred,delta_pred))

class XTFC_S(PIELM):
    def __init__(self,n_nodes,input_size,output_size,length,low_w=-10,high_w=10,low_b=-10,high_b=10,activation_function="tanh",controls=False,physics=False):
        super().__init__(n_nodes,input_size,output_size,length,low_w=-10,high_w=10,low_b=-10,high_b=10,activation_function="tanh",controls=False,physics=False)
        self.length= length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nodes = n_nodes
        self.W = (torch.randn(size=(n_nodes,input_size),dtype=torch.float)*(high_w-low_w)+low_w)
        self.W_t = self.W[:,0]
        self.W_t = self.W_t.reshape(len(self.W_t),1)
        self.W_a =  self.W[:,1:]
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_b-low_b)+low_b)
        self.betas = torch.ones(size=(n_nodes,output_size),requires_grad=True,dtype=torch.float)
        self.controls = controls
        self.physics = physics
        self.z={"max":{0:500,1:1},"min":{0:0,1:0}}
        self.iters=0
        self.p = 0
    def train(self,accuracy, n_iterations,x_train,y_train,lambda_=1):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        count = 0
        error = 100
        self.lambda_ = lambda_
        
        ## Assuming x_train has the following shape, vector of time appended to vector of actions

         ## Assuming x_train has the following shape, vector of time appended to vector of actions
        

        for i in range(x_train.shape[1]):
            z0 = -1
            zf = 1
            t0 = self.z["min"][i]
            tf = self.z["max"][i]
            c = (zf-z0)/(tf-t0)
            x_train[:,i] = z0+c*(x_train[:,i]-t0)
            if i==0:
                self.c = c
        self.x_train = torch.tensor(x_train,dtype=torch.float)
        self.y_train = torch.tensor(y_train,dtype=torch.float)
        h = self.get_h(self.x_train)
        dh = self.c*self.get_dh_t(self.x_train)
        h = torch.vstack((h,dh))
        if self.iters==0:
            self.p = torch.linalg.pinv(torch.matmul(torch.transpose(h,0,1),h))
            self.betas = torch.matmul(torch.matmul(self.p,torch.transpose(h,0,1)),self.y_train)
        else:
            hph = torch.matmul(torch.matmul(h,self.p),torch.transpose(h,0,1))
            hph_inv = torch.linalg.pinv(torch.eye(len(hph))+hph)
            pht = torch.matmul(self.p,torch.transpose(h,0,1))
            hp = torch.matmul(h,self.p)
            update = torch.matmul(torch.matmul(pht,hph_inv),hp)
            self.p = self.p - update
            differential = (self.y_train-torch.matmul(h,self.betas))
            update = torch.matmul(torch.matmul(self.p,torch.transpose(h,0,1)),differential)
            self.betas +=  update 
        y_pred =torch.matmul(h,self.betas)
        self.iters+=1 
        print(((y_pred-self.y_train)**2).mean(),"iter:%i",self.iters)
        

    def predict_jacobian(self,betas):
        
        """""
        For a 2 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1: Then for x we have:
        x(0)       [[1,0]
        xdot(0)    [0,1]
        
        Consider support functions the polynomials: t^0,t^1: Then for theta we have:
        theta(0)       [[1,0]
        thetadot(0)    [0,1]
        """""

        h = self.get_h(self.x_train)
        dh_t = self.get_dh_t(self.x_train)

        bx = betas[0:self.nodes]
        btheta = betas[self.nodes:self.nodes*2]
        
        init_time = self.x_train[0,0].numpy()[0]
        init_h=self.get_h(self.x_train[0,:])
        init_dh=self.get_dh(self.x_train[0,:])
        
        init_x = self.y_train[0,0]
        init_dx = self.speed_x[0,1]
        init_theta = self.y_train[0,2]
        init_dtheta = self.y_train[0,3]
        
        support_function_matrix = np.array([[1,init_time],[0,1]])        
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        free_support_function_matrix = torch.hstack((torch.ones(size=self.x_train.shape),self.x_train))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=self.x_train.shape),torch.ones(size=self.x_train.shape)))
        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(self.x_train),1)
        phi2 = phis[:,1].reshape(len(self.x_train),1)
        
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(self.x_train),1)
        d_phi2 = d_phis[:,1].reshape(len(self.x_train),1)
        
        phi1_h_init = torch.matmul(-phi1,init_h)
        phi1_x_init = phi1*init_x
        phi1_theta_init = phi1*init_theta

        phi2_dh_init = torch.matmul(-phi2,init_dh)
        phi2_dx_init = phi2*init_dx
        phi2_dtheta_init = phi2*init_dtheta

        dphi1_h_init = torch.matmul(-d_phi1,init_h)
        dphi1_x_init = d_phi1*init_x
        dphi1_theta_init = d_phi1*init_theta
        
        dphi2_dh_init = torch.matmul(-d_phi2,init_dh)
        dphi2_dx_init = d_phi2*init_dx
        dphi2_dtheta_init = d_phi2*init_dtheta

        hx = (torch.matmul(h.add(phi1_h_init).add(phi2_dh_init),bx).reshape(self.x_train.shape)\
        .add(phi1_x_init).add(phi2_dx_init/self.c))[:,0]
           
        dhx = (self.c*torch.matmul(dh_t.add(dphi1_h_init).add(dphi2_dh_init),bx).reshape(self.x_train.shape)\
        .add(dphi1_x_init).add(dphi2_dx_init/self.c))[:,0]
       
        htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_dh_init),btheta).reshape(self.x_train.shape)\
        .add(phi1_theta_init).add(phi2_dtheta_init/self.c))[:,0]
     
        dhtheta = (self.c*torch.matmul(dh_t.add(dphi1_h_init).add(dphi2_dh_init),btheta).reshape(self.x_train.shape)\
        .add(dphi1_theta_init).add(dphi2_dtheta_init/self.c))[:,0]

        l_pred_x = self.y_train[:,0]-hx
        l_pred_dhx = self.y_train[:,1] - dhx
        l_pred_theta = self.y_train[:,2]-htheta
        l_pred_dhtheta = self.y_train[:,3] - dhtheta 
        loss= torch.hstack((l_pred_x,l_pred_dhx,l_pred_theta,l_pred_dhtheta))

        return loss
            
    def predict_loss(self,x,y):
        """""
        For a 2 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1: Then for x we have:
        x(0)       [[1,0]
        xdot(0)    [0,1]
        
        Consider support functions the polynomials: t^0,t^1: Then for theta we have:
        theta(0)       [[1,0]
        thetadot(0)    [0,1]
        """""

        h = self.get_h(x)
        dh_t = self.get_dh_t(x)

        bx = self.betas[0:self.nodes]
        btheta = self.betas[self.nodes:self.nodes*2]
        
        init_time = x[0,0].numpy()[0]
        init_h=self.get_h(x[0,:])
        init_dh=self.get_dh(x[0,:])
        
        init_x = y[0,0]
        init_dx = self.speed_x[0,1]
        init_theta = y[0,2]
        init_dtheta = y[0,3]
        
        support_function_matrix = np.array([[1,init_time],[0,1]])        
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        free_support_function_matrix = torch.hstack((torch.ones(size=x.shape),x))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=x.shape),torch.ones(size=x.shape)))
        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(x),1)
        phi2 = phis[:,1].reshape(len(x),1)
        
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(x),1)
        d_phi2 = d_phis[:,1].reshape(len(x),1)
        
        phi1_h_init = torch.matmul(-phi1,init_h)
        phi1_x_init = phi1*init_x
        phi1_theta_init = phi1*init_theta

        phi2_dh_init = torch.matmul(-phi2,init_dh)
        phi2_dx_init = phi2*init_dx
        phi2_dtheta_init = phi2*init_dtheta

        dphi1_h_init = torch.matmul(-d_phi1,init_h)
        dphi1_x_init = d_phi1*init_x
        dphi1_theta_init = d_phi1*init_theta
        
        dphi2_dh_init = torch.matmul(-d_phi2,init_dh)
        dphi2_dx_init = d_phi2*init_dx
        dphi2_dtheta_init = d_phi2*init_dtheta

        hx = (torch.matmul(h.add(phi1_h_init).add(phi2_dh_init),bx).reshape(x.shape)\
        .add(phi1_x_init).add(phi2_dx_init/self.c))[:,0]
           
        dhx = (self.c*torch.matmul(dh_t.add(dphi1_h_init).add(dphi2_dh_init),bx).reshape(x.shape)\
        .add(dphi1_x_init).add(dphi2_dx_init/self.c))[:,0]
       
        htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_dh_init),btheta).reshape(x.shape)\
        .add(phi1_theta_init).add(phi2_dtheta_init/self.c))[:,0]
     
        dhtheta = (self.c*torch.matmul(dh_t.add(dphi1_h_init).add(dphi2_dh_init),btheta).reshape(x.shape)\
        .add(dphi1_theta_init).add(dphi2_dtheta_init/self.c))[:,0]

        l_pred_x = y[:,0]-hx
        l_pred_dhx = y[:,1] - dhx
        l_pred_theta = y[:,2]-htheta
        l_pred_dhtheta = y[:,3] - dhtheta 
        loss= torch.hstack((l_pred_x,l_pred_dhx,l_pred_theta,l_pred_dhtheta))

        return loss
    
    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh_t(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W_t,0,1))
    
    def pred(self,x):


        z0 = -1
        zf = 1
        t0 = x[0,0]
        tf = x[-1,0]
        c = (zf-z0)/(tf-t0)
        x[:,0] = z0+c*(x[:,0]-t0)
        x = torch.tensor(np.array(x),dtype=torch.float)
        
        """""
        For a 2 point constrained problem on x and y  we have: 

        Consider support functions the polynomials: t^0,t^1: Then for x we have:
        x(0)       [[1,0]
        xdot(0)    [0,1]
        
        Consider support functions the polynomials: t^0,t^1: Then for theta we have:
        theta(0)       [[1,0]
        thetadot(0)    [0,1]
        """""

        h = self.get_h(x)
        dh_t = self.get_dh_t(x)

        bx = self.betas[0:self.nodes]
        btheta = self.betas[self.nodes:self.nodes*2]
        
        init_time = x[0,0].numpy()[0]
        init_h=self.get_h(x[0,:])
        init_dh=self.get_dh(x[0,:])
        
        init_x = self.y_train[0,0]
        init_dx = self.speed_x[0,1]
        init_theta = self.y_trainy[0,2]
        init_dtheta = self.y_train[0,3]
        
        support_function_matrix = np.array([[1,init_time],[0,1]])        
        coefficients_matrix = torch.tensor(np.linalg.inv(support_function_matrix),dtype=torch.float)
        free_support_function_matrix = torch.hstack((torch.ones(size=x.shape),x))
        d_free_support_function_matrix = torch.hstack((torch.zeros(size=x.shape),torch.ones(size=x.shape)))
        
        phis = torch.matmul(free_support_function_matrix,coefficients_matrix)
        phi1 = phis[:,0].reshape(len(x),1)
        phi2 = phis[:,1].reshape(len(x),1)
        
        d_phis = torch.matmul(d_free_support_function_matrix,coefficients_matrix)
        d_phi1 = d_phis[:,0].reshape(len(x),1)
        d_phi2 = d_phis[:,1].reshape(len(x),1)
        
        phi1_h_init = torch.matmul(-phi1,init_h)
        phi1_x_init = phi1*init_x
        phi1_theta_init = phi1*init_theta

        phi2_dh_init = torch.matmul(-phi2,init_dh)
        phi2_dx_init = phi2*init_dx
        phi2_dtheta_init = phi2*init_dtheta

        dphi1_h_init = torch.matmul(-d_phi1,init_h)
        dphi1_x_init = d_phi1*init_x
        dphi1_theta_init = d_phi1*init_theta
        
        dphi2_dh_init = torch.matmul(-d_phi2,init_dh)
        dphi2_dx_init = d_phi2*init_dx
        dphi2_dtheta_init = d_phi2*init_dtheta

        hx = (torch.matmul(h.add(phi1_h_init).add(phi2_dh_init),bx).reshape(x.shape)\
        .add(phi1_x_init).add(phi2_dx_init/self.c))[:,0]
           
        dhx = (self.c*torch.matmul(dh_t.add(dphi1_h_init).add(dphi2_dh_init),bx).reshape(x.shape)\
        .add(dphi1_x_init).add(dphi2_dx_init/self.c))[:,0]
       
        htheta = (torch.matmul(h.add(phi1_h_init).add(phi2_dh_init),btheta).reshape(x.shape)\
        .add(phi1_theta_init).add(phi2_dtheta_init/self.c))[:,0]
     
        dhtheta = (self.c*torch.matmul(dh_t.add(dphi1_h_init).add(dphi2_dh_init),btheta).reshape(x.shape)\
        .add(dphi1_theta_init).add(dphi2_dtheta_init/self.c))[:,0]

        l_pred_x = y[:,0]-hx
        l_pred_dhx = y[:,1] - dhx
        l_pred_theta = y[:,2]-htheta
        l_pred_dhtheta = y[:,3] - dhtheta 
        loss= torch.hstack((l_pred_x,l_pred_dhx,l_pred_theta,l_pred_dhtheta))

        
        return torch.vstack((hx,dhx,htheta,dhtheta))

class XTFC_Q(PIELM):
    def __init__(self,n_nodes,input_size,output_size,epsilon,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh",controls=False,physics=False):
        super().__init__(n_nodes,input_size,output_size,length,low_w=-5,high_w=5,low_b=-5,high_b=5,activation_function="tanh",controls=False,physics=False)
        self.length= length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nodes = n_nodes
        self.W = (torch.randn(size=(n_nodes,input_size),dtype=torch.float)*(high_w-low_w)+low_w)
        self.b = (torch.randn(size=(n_nodes,1),dtype=torch.float)*(high_b-low_b)+low_b)
        self.betas = torch.ones(size=(output_size*n_nodes,),requires_grad=True,dtype=torch.float)
        self.controls = controls
        self.physics = physics
        self.epsilon = 1
        self.epsilon_decay = 1e-3
        self.z={"max":{0:2.4,1:10,2:0.2095,3:10},"min":{0:-2.4,1:-10,2:-0.2095,3:-10}}

    def train(self,accuracy, n_iterations,x_train,y_train,lambda_=1):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        count = 0
        error = 100
        self.lambda_ = lambda_
        
        ## Assuming x_train has the following shape, vector of time appended to vector of actions
        for i in range(x_train.shape[1]):
            z0 = -1
            zf = 1
            t0 = x_train[0,i]
            tf = x_train[-1,i]
            c = (zf-z0)/(tf-t0)
            x_train[:,i] = z0+c*(x_train[:,i]-t0)

        self.c = c
        self.x_train = torch.tensor(x_train,dtype=torch.float)
        self.y_train = torch.tensor(y_train,dtype=torch.float)
        print(self.betas.is_cuda)
        print("number of samples:",len(self.x_train))
        h = self.get_h(self.x_train)
        self.betas = torch.matmul(torch.linalg.pinv(h),self.y_train)
        while count < n_iterations:
            
            with torch.no_grad():
                jac = jacobian(self.predict_jacobian,self.betas)
                loss = self.predict_loss(self.x_train,self.y_train)
                pinv_jac = torch.linalg.pinv(jac)
                delta = torch.matmul(pinv_jac,loss)
                self.betas -=delta*0.1
            if count %10==0:
                print(loss.abs().max(dim=0),loss.mean(dim=0))
                print("final loss:",(loss**2).mean())
                print(count)
            count +=1
        

    def predict_jacobian(self,betas):

        h = self.get_h(self.x_train)
        dh_i = self.get_dh_i(self.x_train)
        bq = betas[0:self.nodes]
        hq = torch.matmul(h,bq)
        dhq = self.c*torch.matmul(dh_i,bq)
        l_pred_q = self.y_train-hq
        
        return l_pred_q
            
    def predict_loss(self,x,y):

        h = self.get_h(x)
        dh_i = self.get_dh_i(x)
        bq = self.betas[0:self.nodes]
        hq = torch.matmul(h,bq)
        dhq = self.c*torch.matmul(dh_i,bq)
        l_pred_q = y-hq
        
        return l_pred_q
    
    def get_h(self,x):
        return torch.tanh(torch.add(torch.matmul(x,torch.transpose(self.W,0,1)),torch.transpose(self.b,0,1)))
    def get_dh_i(self,x):
        return torch.mul((1-self.get_h(x)**2),torch.transpose(self.W,0,1))
    
    def pred(self,x):
        
        for i in range(len(x)):
            z0 = -1
            zf = 1
            t0 = self.z["min"][i]
            tf = self.z["max"][i]
            c = (zf-z0)/(tf-t0)
            x[i] = z0+c*(x[i]-t0)
        
        x = torch.tensor(np.array(x),dtype=torch.float)
        h = self.get_h(x)
        dh_i = self.get_dh_i(x)
        bq_1 = self.betas[0:self.nodes]
        bq_2 = self.betas[self.nodes:self.nodes*2]
        
        hq_1 = torch.matmul(h,bq_1)
        hq_2 = torch.matmul(h,bq_2)
        # dhq = self.c*torch.matmul(dh_i,bq)
        return torch.vstack((hq_1,hq_2))
        # return torch.vstack((hq,dhq))