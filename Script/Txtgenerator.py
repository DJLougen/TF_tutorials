import torch
import virtualenv
import tensorflow

virtualenv.run
# Load in models q
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Example text encoding
text = "Im feeling jaded"
encoded_input = tokenizer.encode(text, return_tensors="pt")

# Generate text
output = model.generate(
    encoded_input,
    max_length=1000,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
)


decoded_output = tokenizer.decode(output[0])
print(decoded_output)


# Read in nietzsche
with open("nietzsche txt.txt", "r") as file:
    text = file.read().replace("\n", "")
# Preprocess txt


# Chunk into 1024 token max chunks: GPT2 max is 1024 tokens
chunked_text = []
current_chunk = ""

for sentence in text:
    # Check if adding the current sentence exceeds the maximum token count
    if len(current_chunk.split()) + len(sentence.split()) <= 1024:
        current_chunk += sentence + " "
    else:
        # Append the current chunk to the list and start a new chunk
        chunked_text.append(current_chunk.strip())
        current_chunk = sentence + " "

# Append last chunk
chunked_text.append(current_chunk.strip())

# Encode text
input_nietz = tokenizer.encode(current_chunk, return_tensors="pt")


# Fine tuning
from torch.optim import Adam

# Define optimization algo
optimizer = Adam(model.parameters(), lr=1e-5)  # Learning rate

# Model tuning
model.train()  # Enables training mode
for i in range(1000):  # Epochs
    optimizer.zero_grad()  # Set gradients to zero as pytorch accumulates gradients, zero = fresh start
    outputs = model(input_nietz, labels=input_nietz)  # Labels = retains txt
    loss = outputs.loss  # minmize r
    loss.backward()  # Backprop
    optimizer.step()  # Update model par


# Generate
text = "Who am i?"
encoded_input = tokenizer.encode(text, return_tensors="pt")

# Generate text
output = model.generate(
    encoded_input,
    max_length=1000,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
)


decoded_output = tokenizer.decode(output[0])
print(decoded_output)
