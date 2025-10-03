"""
Fine-tune Pre-trained Phishing Model on ASU Data
=================================================
Starts with cybersectony's model (97.72% accuracy) and improves it!
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import os

class Config:
    # Use the pre-trained model as base!
    MODEL_NAME = "cybersectony/phishing-email-detection-distilbert_v2.1"
    
    MAX_LENGTH = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5  # Lower for fine-tuning
    NUM_EPOCHS = 3  # Fewer epochs since starting from good model
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    OUTPUT_DIR = "./phishing_model_finetuned"
    LOGGING_STEPS = 20
    EVAL_STEPS = 50
    SAVE_STEPS = 50

def load_dataset_for_training():
    try:
        dataset = load_from_disk('./phishing_dataset')
        print("✅ Dataset loaded successfully")
        return dataset
    except:
        print("❌ Error: Dataset not found. Run 1_data_preparation.py first.")
        raise

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples['email_text'],
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_LENGTH
        )
    
    print("Tokenizing dataset...")
    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['label', 'labels']]
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    
    if 'label' in tokenized_dataset['train'].column_names:
        tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    
    return tokenized_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

def train_model():
    print("=" * 70)
    print("FINE-TUNING PRE-TRAINED PHISHING DETECTION MODEL")
    print("=" * 70)
    print(f"\nStarting from: {Config.MODEL_NAME}")
    print("This model already has 97.72% accuracy!")
    print("We'll make it even better with ASU-specific data.\n")
    
    dataset = load_dataset_for_training()
    
    print(f"Loading pre-trained model: {Config.MODEL_NAME}")
    print("(First time will download ~256MB)")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # IMPORTANT: Load pre-trained model but change to 2 classes
    # The original has 4 classes, we need 2 (legitimate vs phishing)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2,  # Our binary classification
        ignore_mismatched_sizes=True  # Allow different output layer
    )
    
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    tokenized_dataset.set_format('torch')
    
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.SAVE_STEPS,
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("\nStarting fine-tuning...")
    print(f"Training on {len(tokenized_dataset['train'])} samples")
    
    train_result = trainer.train()
    
    print("\nSaving fine-tuned model...")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    
    metrics = {
        'base_model': Config.MODEL_NAME,
        'base_model_accuracy': 0.9772,  # Original model's accuracy
        'train_results': {k: float(v) if isinstance(v, (int, float)) else v 
                         for k, v in train_result.metrics.items()},
        'test_results': {k: float(v) if isinstance(v, (int, float)) else v 
                        for k, v in test_results.items()}
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE!")
    print("=" * 70)
    print("\nComparison:")
    print(f"  Original Model: 97.72% accuracy (general phishing)")
    print(f"  Fine-tuned:     {test_results['eval_accuracy']*100:.2f}% accuracy (ASU-specific)")
    print("\nTest Set Results:")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")
    print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {test_results['eval_true_negatives']}")
    print(f"  False Positives: {test_results['eval_false_positives']}")
    print(f"  False Negatives: {test_results['eval_false_negatives']}")
    print(f"  True Positives:  {test_results['eval_true_positives']}")
    print(f"\nModel saved to: {Config.OUTPUT_DIR}")
    
    return trainer, test_results

if __name__ == "__main__":
    trainer, test_results = train_model()
    print("\n✅ Fine-tuning complete!")
    print("\nBenefits of fine-tuning:")
    print("  • Started from 97.72% baseline")
    print("  • Now specialized for ASU phishing")
    print("  • Better at ASU-specific patterns")
    print("  • Faster training (only 3 epochs needed)")