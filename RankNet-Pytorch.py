
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# In[2]:


class RankNet(nn.Module):
    def __init__(self, num_feature):
        super(RankNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear( num_feature, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.output_sig = nn.Sigmoid()

    def forward(self, input_1,input_2):


        s1 = self.model(input_1)
        s2 = self.model(input_2)
        
        out = self.output_sig(s1-s2)

        return out
    
    def predict(self, input_):
        
        s = self.model(input_)
        return s


# In[12]:


n_sample = 30000
n_feature = 300
data1 = torch.rand((n_sample,n_feature))
data2 = torch.rand((n_sample, n_feature))

y = np.random.random(( n_sample,1))
y = y > 0.9
y = 1. * y
y = torch.Tensor(y)

rank_model = RankNet(num_feature=n_feature)
optimizer = torch.optim.Adam(rank_model.parameters())

loss_fun = torch.nn.BCELoss()

rank_model.cuda()
# optimizer.cuda()
loss_fun.cuda()

data1 = data1.cuda()
data2 = data2.cuda()

y = y.cuda()


# In[13]:


epoch = 20000

losses = []

for i in range(epoch):
    
    rank_model.zero_grad()
    
    y_pred = rank_model(data1, data2)
    
    loss = loss_fun(y_pred,y)
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if i % 500 == 0:
    
        print('Epoch{}, loss : {}'.format(i, loss.item()))
    
    


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


x = list(range(10000))


# In[11]:


plt.plot(x, losses)



