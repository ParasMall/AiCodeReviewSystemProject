import json
import torch
import numpy as np
import pandas as pd
import os
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create models directory if it doesn't exist
os.makedirs("models/codebert_model", exist_ok=True)

# Load and preprocess data
def load_data():
    print("Loading data...")
    with open("data/javacode.json", "r") as f:
        data = json.load(f)
    
    # Convert to pandas DataFrame
    buggy_samples = []
    corrected_samples = []
    
    for item in data:
        buggy_samples.append({"code": item["buggy"], "label": 1})  # 1 = buggy
        corrected_samples.append({"code": item["corrected"], "label": 0})  # 0 = correct
    
    df = pd.DataFrame(buggy_samples + corrected_samples)
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    return train_df, val_df

# Tokenize data
def tokenize_data(train_df, val_df):
    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples["code"], padding="max_length", truncation=True, max_length=512)
        return tokenized
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    return train_tokenized, val_tokenized, tokenizer

# Fine-tune CodeBERT model
def train_model(train_tokenized, val_tokenized):
    print("Loading CodeBERT base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=2
    )
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = np.sum(predictions == labels) / len(labels)
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )
    
    print("Training model...")
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save model
    print("Saving model...")
    model.save_pretrained("models/codebert_model")
    
    return model, trainer

# Visualize training results
def plot_training_results(trainer):
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    train_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    
    # Plot validation accuracy
    val_accuracy = [log["eval_accuracy"] for log in trainer.state.log_history if "eval_accuracy" in log]
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracy)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    plt.savefig("codebert_training_results.png")
    plt.close()

if __name__ == "__main__":
    print("Starting CodeBERT fine-tuning for bug detection...")
    train_df, val_df = load_data()
    train_tokenized, val_tokenized, tokenizer = tokenize_data(train_df, val_df)
    tokenizer.save_pretrained("models/codebert_model")
    model, trainer = train_model(train_tokenized, val_tokenized)
    plot_training_results(trainer)
    print("CodeBERT model training complete!") 