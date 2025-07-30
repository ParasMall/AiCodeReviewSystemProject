import json
import torch
import numpy as np
import pandas as pd
import os
import random
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt

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
os.makedirs("models/codet5_model", exist_ok=True)

# Load and preprocess data
def load_data():
    print("Loading data...")
    with open("data/javacode.json", "r") as f:
        data = json.load(f)
    
    # Create dataset with source (buggy) and target (corrected) code
    dataset = []
    for item in data:
        dataset.append({
            "source": item["buggy"],  # Input: buggy code
            "target": item["corrected"]  # Output: corrected code
        })
    
    df = pd.DataFrame(dataset)
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    return train_df, val_df

# Tokenize data
def tokenize_data(train_df, val_df):
    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["source"], padding="max_length", truncation=True, max_length=512)
        
        # Set up the tokenizer for targets
        labels = tokenizer(text_target=examples["target"], padding="max_length", truncation=True, max_length=512)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    train_tokenized = train_dataset.map(preprocess_function, batched=True)
    val_tokenized = val_dataset.map(preprocess_function, batched=True)
    
    return train_tokenized, val_tokenized, tokenizer

# Fine-tune CodeT5 model
def train_model(train_tokenized, val_tokenized, tokenizer):
    print("Loading CodeT5 base model...")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    
    # Define evaluation metrics
    bleu = evaluate.load("bleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode predictions
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU score
        result = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        
        return {"bleu": result["bleu"]}
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        do_eval=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="bleu"
    )
    
    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print("Training model...")
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save model
    print("Saving model...")
    model.save_pretrained("models/codet5_model")
    
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
    
    # Plot validation BLEU
    val_bleu = [log["eval_bleu"] for log in trainer.state.log_history if "eval_bleu" in log]
    plt.subplot(1, 2, 2)
    plt.plot(val_bleu)
    plt.title("Validation BLEU Score")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    
    plt.tight_layout()
    plt.savefig("codet5_training_results.png")
    plt.close()

if __name__ == "__main__":
    print("Starting CodeT5 fine-tuning for bug fixing...")
    train_df, val_df = load_data()
    train_tokenized, val_tokenized, tokenizer = tokenize_data(train_df, val_df)
    tokenizer.save_pretrained("models/codet5_model")
    model, trainer = train_model(train_tokenized, val_tokenized, tokenizer)
    plot_training_results(trainer)
    print("CodeT5 model training complete!") 