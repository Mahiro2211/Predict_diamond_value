#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
import torch
import matplotlib
from pandas import Series , DataFrame
import os
import torch.nn as nn
import numpy as np


# In[161]:


f = pd.read_excel('附件2.xlsx')
header = pd.read_excel('附件2.xlsx' , nrows=0)


# In[162]:


f.head()


# In[163]:


carat = [i for _ , i in f['carat'].items()]
depth = [i for _ , i in f['depth'].items()]
tabel = [i for _ , i in f['table'].items()]
x = [i for _ , i in f['x'].items()]
y = [i for _ , i in f['y'].items()]
price = [i for _, i in f['price'].items()]


# In[164]:


headerlist = list(header)
#print(headerlist)
# 读取有多少颜色
color = list(dict(f['color']).values())
binary_color = f['color']
color = set(color)
#print(color)


# * 说明以下，这里我注意到，颜色是一个中性的参数，也就是说如果给颜色分三六九等会导致模型准确度不好，你自己想也是这样
# * 所以我们把颜色变量二进制化处理

# In[165]:


df = pd.DataFrame(binary_color)
#print(df)
color_dum = pd.get_dummies(df , prefix='color')#二进制化后的颜色
#print(color_dum)


# In[166]:


# 给清晰度作映射
clarity = list(dict(f['clarity']).values())
set_cl = set(clarity)
#print(set_cl)
list_cl = ['I1' , 'SI1' , 'SI2' , 'VS1' , 'VS2' , 'VVS1' , 'VVS2' , 'IF']
dict_cl = {k : v for k , v in enumerate(list_cl)}
#print(dict_cl)
dict_cl = {v : k for k , v in dict_cl.items()}
num_cl = []
for i in range(len(clarity)):
    num_cl.append(dict_cl[clarity[i]] + 1)


# In[167]:


#给颜色和品质做一个映射，把他们变成数字方便我们处理
# 品质确实分369等
cut_list = [v for k , v in f['cut'].items()]
cut = ['Fair' , 'Good' , 'Very Good' , 'Premium' , 'Ideal']
res_cut = dict(enumerate(cut))
res_cut = {v : k for k , v in res_cut.items()}

labels=[]
for i in range(len(cut_list)):
    labels.append(res_cut[cut_list[i]] + 1)
set_label = set(labels)
#print(labels)


# In[168]:


color_d = color_dum['color_D'].values.tolist()
color_e = color_dum['color_E'].values.tolist()
color_f = color_dum['color_F'].values.tolist()
color_g = color_dum['color_G'].values.tolist()
color_h = color_dum['color_H'].values.tolist()
color_i = color_dum['color_I'].values.tolist()
color_j = color_dum['color_J'].values.tolist()


# In[169]:


carat = np.array(carat)
depth = np.array(depth)
x = np.array(x)
y = np.array(y)
price = np.array(price)
labels = np.array(labels)
color_d = np.array(color_d)
color_e = np.array(color_e)
color_f = np.array(color_f)
color_g = np.array(color_g)
color_h = np.array(color_h)
color_i = np.array(color_i)
color_j = np.array(color_j)


# In[170]:


carat = torch.tensor(carat)
depth = torch.tensor(depth)
x = torch.tensor(x)
y = torch.tensor(y)
lb_price = torch.tensor(price)
labels = torch.tensor(labels)
color_d = torch.tensor(color_d)
color_e = torch.tensor(color_e)
color_f = torch.tensor(color_f)
color_g = torch.tensor(color_g)
color_h = torch.tensor(color_h)
color_i = torch.tensor(color_i)
color_j = torch.tensor(color_j)


# In[171]:


#print(x)


# In[172]:


data_need_pred = torch.stack([carat, depth, x, y, labels, color_d,
                     color_e, color_f, color_g, color_h,
                     color_i, color_j], dim=1)
#这样我们就得到了一个（N ， 12）大小的张量


# In[ ]:


# In[ ]:




