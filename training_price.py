#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from data_changed import * # 里头的items是我们用来训练的
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import joblib
from torch.utils.data import Dataset


# In[2]:




# In[3]:


items_mean = items.mean(dim=0)
items_std = items.std(dim=0)
items_normalized = (items - items_mean)/items_std
items_normalized
x = items_normalized[:, :-1]  # 特征张量
y = items_normalized[:, -1]   # 标签张量
train_size = int(0.8 * len(items_normalized))
train_x, test_x = x[:train_size], x[train_size:]
train_y, test_y = y[:train_size], y[train_size:]
scaler = (items_mean, items_std)
joblib.dump(scaler , "diamond_price_scaler.pkl")
train_x.size() , test_x.size() , train_y.size()


# In[12]:


class DiamondPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).double()  # 将权重张量和偏置张量的数据类型改为torch.float64
        self.fc2 = nn.Linear(hidden_dim , 512).double()
        self.fc3 = nn.Linear(512 , hidden_dim).double()
        self.fc4 = nn.Linear(hidden_dim ,output_dim).double()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 初始化模型
input_dim = 12
hidden_dim = 64
output_dim = 1
model = DiamondPricePredictor(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_x)
    loss = criterion(outputs, train_y.view(-1, 1))
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印损失值
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 在测试集上评估模型
with torch.no_grad():
    test_outputs = model(test_x)
    test_loss = criterion(test_outputs, test_y.view(-1, 1))
    print('Test Loss: {:.4f}'.format(test_loss.item()))
    


# In[15]:


torch.save(model, 'diamond_price.pt')
from data_need_pre import *
with torch.no_grad():
    test_outputs = model(test_x)
    test_loss = criterion(test_outputs, test_y.view(-1, 1))
    print('Test Loss: {:.4f}'.format(test_loss.item()))


# In[43]:


# 加载模型和归一化器
model = torch.load('diamond_price.pt')
scaler = joblib.load('diamond_price_scaler.pkl')

# 准备输入数据
input_data = data_need_pred

# 计算输入数据的均值和标准差
input_data_mean = input_data.mean(dim=0, keepdim=True)
input_data_std = input_data.std(dim=0, keepdim=True)

# 对输入数据进行归一化处理
input_data_normalized = (input_data - input_data_mean) / input_data_std

# 将输入数据的数据类型转换为与模型权重相同的数据类型
input_data_normalized = input_data_normalized.to(model.fc1.weight.dtype)

# 预测归一化的价格
predicted_price_normalized = model(input_data_normalized)

# 反归一化输出价格
predicted_price = predicted_price_normalized * scaler[1][-1] + scaler[0][-1]

# 打印预测的价格
predicted_price_np = predicted_price.detach().numpy()
for i in range(len(predicted_price_np)):
    print('Predicted price {}: ${:.2f}'.format(i+1, float(predicted_price_np[i])))
print(type(predicted_price_np))
predicted_price_np = predicted_price_np.sum(axis=1)
predicted_price_np = predicted_price_np.tolist()
print(predicted_price_np)


# In[45]:


predicted_price_np = predicted_price_np
predicted_price_np_rounded = [round(x) for x in predicted_price_np]

f['price'] = {k:predicted_price_np[k] for k in range(len(f['price']))}
# 将数据框写入到Excel文件中（使用XlsxWriter作为引擎）
writer = pd.ExcelWriter('输出.xlsx', engine='xlsxwriter')
f.to_excel(writer, index=False, sheet_name='Sheet1')

# 获取workbook对象
workbook = writer.book

# 关闭Excel写入对象并保存文件
writer.close()
workbook.close()


# In[33]:




# In[ ]:





# In[ ]:




