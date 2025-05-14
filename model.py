from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

#read in info from info.csv
df = pd.read("data/info.csv")

#split data into training and validation.
#test_size = 0.1 --> 10% is validation
#stratify --> makes sure equal representation of livberal and conservative posts looked at
#random_state sets seed so the split is the same every time 
#train_df, val_df saves this split into two data frames.
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["Political Lean"], random_state=42)

# establish base model that we are going to LoRA on

model_id = "google/gemma-3-1b-it"  # or "google/gemma-1.1-7b-it" if you have a strong GPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples['Text'], padding='max_length', truncation=True)

# Tokenize dataset splits so readable to computer
train_dataset = train_df.map(tokenize_function, batched=True)
val_dataset = val_df.map(tokenize_function, batched=True)

#done for now. Next I will pair labels and political lean via a map so that the model can predict political sentiment and guess responses
# Then I will install LoRA and implement it using peft.  

#testing-----ignore----------
#prompt = "What are the main differences between liberal and conservative ideologies?"
#nputs = tokenizer(prompt, return_tensors="pt")
# Generate text
#outputs = model.generate(**inputs, max_new_tokens=400)
#response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(response)

#--------------#