import pandas as pd

# Load your dataset (example with a CSV file)
df = pd.read_csv("courses1.csv")

# Example of dataset structure
print(df.head())



from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Example function to create the input prompt
def create_prompt(title, description, curriculum):
    # Define the prompt structure
    prompt = f"Course Title: {title}\nDescription: {description}\nCurriculum: {curriculum}\n"
    prompt += "Summarize this course or recommend it based on the description."
    return prompt

# Dataset class for handling course data
class CourseDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        title = self.df.iloc[idx]['title']
        description = self.df.iloc[idx]['description']
        curriculum = self.df.iloc[idx]['curriculum']
        
        # Generate prompt
        input_text = create_prompt(title, description, curriculum)
        
        # Tokenize input and output
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Set the target output as a summary or recommendation task
        labels = self.tokenizer.encode_plus(
            "Summarize this course",
            max_length=100,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

# Initialize Dataset and DataLoader
dataset = CourseDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):  # Train for 3 epochs
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print("Training Complete!")





# Test the fine-tuned model with a new prompt
def generate_course_summary(model, tokenizer, title, description, curriculum, max_length=100):
    model.eval()
    input_text = create_prompt(title, description, curriculum)
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = input_ids.to(device)
    
    # Generate output (summary or recommendation)
    summary_ids = model.generate(input_ids=input_ids, max_length=max_length, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Example usage
title = "Introduction to Deep Learning"
description = "This course covers the basics of neural networks and deep learning."
curriculum = "Week 1: Neural Networks, Week 2: Backpropagation, Week 3: CNNs"

summary = generate_course_summary(model, tokenizer, title, description, curriculum)
print(f"Generated Summary: {summary}")





