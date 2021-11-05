#!/usr/bin/env python
# coding: utf-8

# In[1]:


#When converting to Python file, change display function to print function


# In[2]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
from torch.utils.data import Dataset 
import random
import time
import datetime
import random
from transformers import GPT2LMHeadModel, GPT2Config
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import matplotlib.pyplot as plt


# In[3]:


BATCH_SIZE = 16


# In[4]:


import pandas as pd

data = pd.read_csv('../Dataset/0OYkPK', sep=",", header=None)
data.columns = data.iloc[0]
data = data[1:]
data['full'] = data['sentence1']+ " " + data['sentence2']+ " " + data['sentence3']+ " " + data['sentence4']+ " " + data['sentence5']
data['input'] = data['sentence1']+ " " + data['sentence2']+ " " + data['sentence3']+ " " + data['sentence4']
data


# In[5]:


val_data = pd.read_csv('../Dataset/XWjas1', sep=",", header=None)
val_data.columns = val_data.iloc[0]
val_data = val_data[1:]
#val_data['InputSentence5'] = val_data['RandomFifthSentenceQuiz1']
#val_data['InputSentence5'] = val_data['RandomFifthSentenceQuiz1']
val_data['InputSentence5']  = np.where(val_data['AnswerRightEnding']== '1', val_data['RandomFifthSentenceQuiz1'], val_data['RandomFifthSentenceQuiz2'])
val_data['full'] = val_data['InputSentence1']+ " " + val_data['InputSentence2']+ " " + val_data['InputSentence3']+ " " + val_data['InputSentence4']+ " " + val_data['InputSentence5']
val_data['input'] = val_data['InputSentence1']+ " " + val_data['InputSentence2']+ " " + val_data['InputSentence3']+ " " + val_data['InputSentence4']
val_data


# In[6]:


with torch.cuda.device('cuda:1'):
    torch.cuda.empty_cache()

all_sentences = [x for x in data.full]#[:1000]]

val_sentences = [x for x in val_data.input]#[:1000]]


# In[7]:


all_sentences[0:10]


# In[8]:


from transformers import GPT2Tokenizer
#get pretrained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<sos>', pad_token='<pad>', padding_side = 'left')


# In[9]:


max_len = int(np.max([len(tokenizer.encode(s)) for s in all_sentences]))
print(f"max_len {max_len}")

max_val = int(np.max([len(tokenizer.encode(s)) for s in val_sentences]))
print(f"max_val {max_val}")


# In[10]:


def tokenize_seq(sent,tokenizer,max_length):
    return tokenizer('<sos>'+ sent , truncation=True, max_length=max_length, padding="max_length")

class JapanDataset(Dataset):

    def __init__(self, sentences, tokenizer, gpt2_type="gpt2", max_length=max_len):

        self.tokenizer = tokenizer 
        self.input_ids = []
        self.attn_masks = []

        for sentence in sentences:      
            encodings = tokenize_seq(sentence,tokenizer,max_length)

            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]   

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


# In[11]:


import gc
gc.collect()


# In[12]:


#create an instance of Dataset
train_set = JapanDataset(all_sentences, tokenizer, max_length=max_len)
val_set = JapanDataset(val_sentences, tokenizer, max_length=max_val)


#train_set, val_set = random_split(dataset, [train_size, val_size])
#print("train_size :",train_size)
#print("val_size   :",val_size)

gc.collect()


# In[13]:


print(train_set[0])


# In[14]:


#define dataloaders
train_dataloader = DataLoader(train_set,  sampler = RandomSampler(train_set), batch_size = BATCH_SIZE)
validation_dataloader = DataLoader(val_set, sampler = SequentialSampler(val_set), batch_size = BATCH_SIZE )


# In[15]:


# Create default config
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
# Load pretrained gpt2
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))

# Create device
device = torch.device("cuda:1")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr = 0.00005)
model = model.to(device)


# In[16]:


from tqdm import tqdm

#call model with a batch of input
def process_one_batch(batch):
    b_input_ids = batch[0].to(device)
    b_labels = batch[0].to(device)
    b_masks = batch[1].to(device)
    outputs  = model(b_input_ids,  attention_mask = b_masks,labels=b_labels)
    return outputs

#call model with a batch of input
def output_one_batch(batch):
    b_input_ids = batch[0].to(device)
    b_labels = batch[0].to(device)
    b_masks = batch[1].to(device)
    outputs  = model(b_input_ids,  num_beams=5 ,  attention_mask = b_masks,labels=b_labels)
    return outputs

#do one epoch for training
def train_epoch():
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):

        model.zero_grad()        
        outputs = process_one_batch( batch)
        loss = outputs[0]  
        batch_loss = loss.item()
        total_train_loss += batch_loss

        loss.backward()
        optimizer.step()


    avg_train_loss = total_train_loss / len(train_dataloader)  
    print("avg_train_loss",avg_train_loss)  
    elapsed_time = format_time(time.time() - t0)
    print("elapsed time for 1 training epoch : ",elapsed_time)
    return avg_train_loss

#do one epoch for eval
def eval_epoch():
    t0 = time.time()
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:            

        with torch.no_grad():        
            outputs = process_one_batch(batch)
            loss = outputs[0]              
            batch_loss = loss.item()
            total_eval_loss += batch_loss         

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("avg_val_loss",avg_val_loss) 
    elapsed_time = format_time(time.time() - t0)
    print("elapsed time for 1 eval epoch : ",elapsed_time)
    return avg_val_loss
    

#do one epoch for eval
def save_results( num_batches = 15, iter = 0):
    indexes_list = []
    inputs_list = []
    predicted_list = []
    expected_list = []
        
    #for i in tqdm(range(num_examples)):
    for i, batch in enumerate(tqdm(validation_dataloader)):
        if(num_batches != None and i>num_batches):
            break
        # Story is:
        #input_ids = tokenizer(val_data.input.iloc[i], return_tensors='pt')
        #input_ids.to(device)
        b_input_ids = batch[0].to(device)
        
        greedy_output = model.generate(
                b_input_ids,  #check stars   
                num_beams=5 ,
                return_dict_in_generate=True, 
                output_scores=True, 
                max_length=150
                )
        
        print(greedy_output['sequences'].shape)
        output = tokenizer.batch_decode(greedy_output['sequences'])
        len_input = len(val_data.input.iloc[i])
        #output = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        b_outputs = []
        
        for b_idx in range(BATCH_SIZE):
            if(len(output) <= b_idx):
                break
            idx = BATCH_SIZE * i + b_idx
            end_words = val_data.InputSentence4
            in_words = val_data.input.iloc[idx]
            #if(len(output) <= b_idx):
            #    break
            #b_out = output[b_idx]
            #pred_sent = b_out[b_out.index('<sos>') + len(in_words) + 13:]
            #if(pred_sent.find('.') != -1):
            #    pred_sent = pred_sent[:pred_sent.index('.')+1]
            #else:
            #    pred_sent = pred_sent
            
            indexes_list.append(idx)
            inputs_list.append(val_data.input.iloc[idx])
            predicted_list.append(output[b_idx])
            expected_list.append(val_data.InputSentence5.iloc[idx])
            
    outputs = pd.DataFrame()
    outputs['inputs'] = inputs_list
    outputs['predicted'] = predicted_list
    outputs['expected'] = expected_list
    
    print(outputs[:5])

            
    
    
    data = {}
    data['ex'] = []
    for i in range(len(indexes_list)):
        data['ex'].append({
            'idx': indexes_list[i],
            'input': inputs_list[i],
            'prediction': predicted_list[i],
            'expected': expected_list[i]

        })

    with open('test_cases'+ str(iter)+'.json', 'w') as outfile:
        json.dump(data, outfile)


# In[17]:


save_results(1)


# In[18]:


train_loss = []
val_loss = []
print('initial val loss ', eval_epoch())
for i in range(15):
    train_loss.append(train_epoch())
    val_loss.append(eval_epoch())
    if(i<7):
        save_results(None, i)
    else:
        save_results(100, i)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.show()
    plt.savefig('losses.png')
    print(train_loss)
    print(val_loss)

save_results(None)


# ##### 

# In[ ]:




