#!/usr/bin/env python
# coding: utf-8

# In[74]:



import json
import os


file_path = os.path.abspath(os.path.dirname(__file__))
print(file_path)
# Opening JSON file
f = open(file_path + '/test_cases/test_cases.json',)


# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
idx = []
inputs = []
outputs = []
expected = []

for i in data['ex']:
    #print(i)
    
    idx.append(i['idx'])
    inputs.append(i['input'])
    outputs.append(i['prediction'])
    expected.append(i['expected'])
 
# Closing file
f.close()


# In[75]:


import pandas as pd
df = pd.DataFrame()
df['idx'] = idx
df['input'] = inputs
df['output'] = outputs
df['expected'] = expected
df = df.iloc[:700]
df

# In[80]:


output_lens = []
for i in range(5):
    output_lens.append(len(df.output.apply(lambda x: x.split()[i] if len(x.split())>i else None).unique()))
    
expected_lens = []
for i in range(5):
    expected_lens.append(len(df.expected.apply(lambda x: x.split()[i] if len(x.split())>i else None).unique()))


# In[81]:


print('\nPrinting the number of unique words at the first 5 word positions')
print('predicted sentence ', output_lens, '->mean: ', sum(output_lens)/5)


# In[82]:


print('expected sentence ', expected_lens, '->mean: ',sum(expected_lens)/5)
print()

# In[ ]:




