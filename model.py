from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments 
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, LoraConfig, PeftModel
from sklearn.model_selection import train_test_split
import os
import torch

#read in info from info.csv
df = pd.read_csv("data/info.csv")

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

lora_config = LoraConfig(
    r=4, #As bigger the R bigger the parameters to train.
    lora_alpha=2, # a scaling factor that adjusts the magnitude of the weight matrix. Usually set to 1
    target_modules=["q_proj", "v_proj"], #You can obtain a list of target modules in the URL above.
    lora_dropout=0.05, #Helps to avoid Overfitting.
    bias="lora_only", # this specifies if the bias parameter should be trained.
    task_type="CAUSAL_LM"
)

label2id = {"Conservative": 0, "Liberal": 1}
def tokenize_function(examples):
    tokens = tokenizer(
        examples["Title"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokens["labels"] = label2id[examples["Political Lean"]]

    return tokens

# Tokenize dataset splits so readable to computer

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)


peft_model = get_peft_model(model, lora_config)

temp_output_dir = os.path.join('./', "peft_lab_outputs")

training_args = TrainingArguments(
    output_dir=temp_output_dir,
    auto_find_batch_size=True, # Find a correct bvatch size that fits the size of Data.
    learning_rate= 3e-2, # Higher learning rate than full fine-tuning.
    num_train_epochs=2,
    use_cpu=True
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

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