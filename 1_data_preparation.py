"""
Phishing Email Detection - Data Preparation
============================================
This script loads and prepares datasets for training a phishing detection model.
We'll use multiple public datasets for robust training.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import re

# ============================================
# 1. LOAD DATASETS FROM HUGGING FACE
# ============================================

def load_phishing_datasets():
    """
    Load multiple phishing datasets and combine them.
    Using publicly available datasets from Hugging Face.
    """
    
    print("Loading datasets...")
    
    # Dataset 1: Email phishing dataset
    # This is a common phishing email dataset
    try:
        ds1 = load_dataset("ealvaradob/phishing-dataset")
        df1 = pd.DataFrame(ds1['train'])
        df1 = df1.rename(columns={'text': 'email_text', 'label': 'label'})
        print(f"Dataset 1 loaded: {len(df1)} samples")
    except:
        print("Dataset 1 not available, skipping...")
        df1 = pd.DataFrame()
    
    # Dataset 2: Alternative phishing dataset
    try:
        ds2 = load_dataset("elozano/phishing_email")
        df2 = pd.DataFrame(ds2['train'])
        if 'email' in df2.columns:
            df2 = df2.rename(columns={'email': 'email_text'})
        print(f"Dataset 2 loaded: {len(df2)} samples")
    except:
        print("Dataset 2 not available, creating synthetic data...")
        df2 = create_synthetic_dataset()
    
    return df1, df2


def create_synthetic_dataset():
    """
    Create a synthetic dataset with phishing and legitimate emails.
    This ensures we have training data even if datasets fail to load.
    """
    
    # Phishing email templates (based on common patterns)
    phishing_emails = [
        "URGENT: Your account will be suspended! Click here to verify: http://suspicious-link.com",
        "Congratulations! You've won $1,000,000! Claim now at: http://fake-lottery.com",
        "Your password expires today. Reset immediately: http://phishing-site.com/reset",
        "Security Alert: Unusual activity detected. Verify your identity: http://fake-bank.com",
        "Your package is pending. Update delivery info: http://scam-delivery.com",
        "IRS Tax Refund: You are eligible for $2,500 refund. Click to claim: http://fake-irs.com",
        "Your Netflix subscription has failed. Update payment: http://netflix-scam.com",
        "Action Required: Verify your email within 24 hours: http://verify-scam.com",
        "Your Amazon order #12345 has been cancelled. Review here: http://amazon-fake.com",
        "Microsoft Security: Your PC is infected. Download fix: http://virus-scam.com",
        "PayPal: Confirm your account to avoid suspension: http://paypal-verify.tk",
        "Bank Alert: Suspicious transaction detected: http://secure-bank.com.fake.com",
        "Apple ID: Your account has been locked: http://appleid-unlock.net",
        "Google: Verify your identity now: http://google-verify.co",
        "Facebook: Someone tried to access your account: http://fb-security.com",
    ]
    
    # Legitimate email templates
    legitimate_emails = [
        "Hi team, please review the attached quarterly report for our meeting tomorrow.",
        "Thank you for your order. Your confirmation number is 123456. Expected delivery: Friday.",
        "Reminder: Department meeting scheduled for 2 PM in Conference Room B.",
        "Your password was successfully changed on 2024-10-01. If this wasn't you, contact support.",
        "Welcome to ASU! Here's your student orientation schedule for next week.",
        "Your library book 'Introduction to Machine Learning' is due on October 15th.",
        "Course registration for Spring 2025 opens on November 1st at 8 AM.",
        "Your grade for CSE 445 has been posted. Log in to My ASU to view.",
        "Campus shuttle schedule updated. New routes available starting next Monday.",
        "Financial aid disbursement will occur on October 10th. Check your account.",
        "Office hours this week: Tuesday 2-4 PM, Thursday 3-5 PM in BYENG 120.",
        "Project deadline extended to Friday. Updated rubric posted on Canvas.",
        "Your tuition payment has been processed. Receipt available in My ASU.",
        "Career fair on October 20th. Bring resumes and dress professionally.",
        "Study group meeting tonight at 7 PM in the library, Room 301.",
    ]
    
    # Create expanded dataset with variations
    phishing_data = []
    for email in phishing_emails:
        for i in range(20):  # Create variations
            phishing_data.append({
                'email_text': email + f" [Ref: {np.random.randint(10000, 99999)}]",
                'label': 1
            })
    
    legitimate_data = []
    for email in legitimate_emails:
        for i in range(20):
            legitimate_data.append({
                'email_text': email + f" ID: {np.random.randint(1000, 9999)}",
                'label': 0
            })
    
    df = pd.DataFrame(phishing_data + legitimate_data)
    return df


def clean_email_text(text):
    """
    Clean and preprocess email text.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()


def prepare_dataset():
    """
    Main function to prepare the complete dataset.
    """
    
    # Load datasets
    df1, df2 = load_phishing_datasets()
    
    # Combine datasets
    if not df1.empty and not df2.empty:
        df = pd.concat([df1, df2], ignore_index=True)
    elif not df1.empty:
        df = df1
    elif not df2.empty:
        df = df2
    else:
        df = create_synthetic_dataset()
    
    # Ensure required columns exist
    if 'email_text' not in df.columns:
        if 'text' in df.columns:
            df = df.rename(columns={'text': 'email_text'})
        elif 'email' in df.columns:
            df = df.rename(columns={'email': 'email_text'})
    
    # Clean text
    df['email_text'] = df['email_text'].apply(clean_email_text)
    
    # Remove empty texts
    df = df[df['email_text'].str.len() > 10]
    
    # Ensure binary labels (0 = legitimate, 1 = phishing)
    df['label'] = df['label'].astype(int)
    
    # Balance dataset if needed
    phishing_count = (df['label'] == 1).sum()
    legit_count = (df['label'] == 0).sum()
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Phishing emails: {phishing_count}")
    print(f"Legitimate emails: {legit_count}")
    
    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"\nSplit Statistics:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    # Convert to Hugging Face Dataset format
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })
    
    # Save to disk
    dataset.save_to_disk('./phishing_dataset')
    print("\nDataset saved to ./phishing_dataset")
    
    return dataset


if __name__ == "__main__":
    dataset = prepare_dataset()
    print("\nSample emails:")
    print("\nPhishing example:")
    phishing_example = [ex for ex in dataset['train'] if ex['label'] == 1][0]
    print(phishing_example['email_text'][:200])
    print("\nLegitimate example:")
    legit_example = [ex for ex in dataset['train'] if ex['label'] == 0][0]
    print(legit_example['email_text'][:200])