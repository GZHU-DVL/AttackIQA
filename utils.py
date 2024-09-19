import torch
import torch.nn as nn
import numpy as np
import os
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.para = torch.nn.Parameter(torch.ones([3, 1]))

        # 用于存储optim的相关变量
        self.para_pre = self.para

        # SGD相关
        self.gradient_pre = 0
        self.buffer = 0  # buffer, 相当于存速度

        # Adam相关
        self.moment = 0
        self.velocity = 0
        self.velocity_max = torch.zeros([3,1])

        # RMSprop相关
        self.gradient_ave = 0

    def calculate(self, x, y):

        pred = torch.mm(x.to(self.para), self.para)
        loss = torch.mean(pred - y)

def matrix_adam(idx, model, x, y, lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.1, amsgrad=False):

    beta1, beta2 = betas

    pred, loss = model.calculate(x, y)
    # 得到梯度
    gradient = torch.mean(x, dim=0).reshape([3, 1])

    if weight_decay != 0:
        gradient = gradient + weight_decay * model.para_pre

    model.moment = beta1 * model.moment + (1-beta1) * gradient
    model.velocity = beta2 * model.velocity + (1 - beta2) * (gradient * gradient)

    moment = model.moment / (1 - (beta1 ** (idx + 1)))
    velocity = model.velocity / (1 - (beta2 ** (idx + 1)))

    if amsgrad:
        model.velocity_max = torch.maximum(model.velocity_max, model.velocity)
        model.para.data = model.para- lr * moment / \
                          (torch.sqrt(model.velocity_max / (1 - (beta2 ** (idx + 1)))) + eps)
    else:
        model.para.data = model.para - lr * moment / (torch.sqrt(velocity) + eps)

    return model.para

import math

alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999            #initialize the values of the parameters
epsilon = 1e-8
def func(x):
  return x*x -4*x + 4
def grad_func(x):         #calculates the gradient
  return 2*x - 4
theta_0 = 0           #initialize the vector
m_t = 0 
v_t = 0 
t = 0

while (1):          #till it gets converged
  t+=1
  g_t = grad_func(theta_0)    #computes the gradient of the stochastic function
  m_t = beta_1*m_t + (1-beta_1)*g_t #updates the moving averages of the gradient
  v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) #updates the moving averages of the squared gradient
  m_cap = m_t/(1-(beta_1**t))   #calculates the bias-corrected estimates
  v_cap = v_t/(1-(beta_2**t))   #calculates the bias-corrected estimates
  theta_0_prev = theta_0                
  theta_0 = theta_0 - (alpha*m_cap)/(math.sqrt(v_cap)+epsilon)  #updates the parameters
  if(theta_0 == theta_0_prev):    #checks if it is converged or not
    break

  


