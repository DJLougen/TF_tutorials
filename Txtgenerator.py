import torch
import virtualenv
import tensorflow
virtualenv.run
#Load in models 
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

#Example text encoding 
text = "Im feeling jaded"
encoded_input = tokenizer.encode(text, return_tensors='pt')

#Generate text 
output = model.generate(encoded_input, max_length = 10000, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)


decoded_output = tokenizer.decode(output[0])
print(decoded_output)