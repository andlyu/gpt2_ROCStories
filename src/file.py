import pandas as pd

data = pd.read_csv('../Dataset/0OYkPK', sep=",", header=None)
data.columns = data.iloc[0]
data = data[1:]
data['full'] = data['sentence1']+ " " + data['sentence2']+ " " + data['sentence3']+ " " + data['sentence4']+ " " + data['sentence5']
data['input'] = data['sentence1']+ " " + data['sentence2']+ " " + data['sentence3']+ " " + data['sentence4']

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')
# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

for i in range(5):
    # Story is:
    input_ids = tokenizer.encode(data.input.iloc[i], return_tensors='tf')
    greedy_output = model.generate(input_ids, max_length=50)
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
    print()

    