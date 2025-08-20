import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import torch._dynamo

# Disable torch compilation to avoid Triton issues
torch._dynamo.config.disable = True
os.environ['TORCH_COMPILE_DISABLE'] = '1'
torch._dynamo.config.suppress_errors = True

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and prepare the data
df = pd.read_excel(r"C:\Users\doganeren.ozer\Desktop\twitter scraping\classification_and_preprocess_code\karistirilmis_sms_dataset.xlsx")

# Convert spam labels to numerical values
label_map = {'ham': 0, 'spam': 1}
df['label_numeric'] = df['label'].map(label_map)

# Basic text preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

# Apply preprocessing
df['clean_message'] = df['message'].apply(preprocess_text)

# Remove any rows with missing labels or empty messages
df = df.dropna(subset=['label_numeric'])
df = df[df['clean_message'].str.len() > 0]

print(f"Dataset shape after preprocessing: {df.shape}")
print(f"Label distribution:")
print(df['label'].value_counts())

# Split the data
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['clean_message'].values, df['label_numeric'].values, test_size=0.3, random_state=42, stratify=df['label_numeric']
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train set size: {len(train_texts)}")
print(f"Validation set size: {len(val_texts)}")
print(f"Test set size: {len(test_texts)}")

# Custom Dataset class
class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-uncased",
    num_labels=2  # Binary classification: ham vs spam
)

# Create datasets and dataloaders
batch_size = 16

train_dataset = SMSDataset(train_texts, train_labels, tokenizer)
val_dataset = SMSDataset(val_texts, val_labels, tokenizer)
test_dataset = SMSDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 2

# Create output directory
output_dir = 'sms_spam_detection'
os.makedirs(output_dir, exist_ok=True)

# Training loop
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Calculate accuracy
        preds = torch.argmax(outputs.logits, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    actual_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions, average='weighted')
    return predictions, actual_labels, avg_loss, accuracy, f1

# Training and evaluation with best model saving
print("Starting training...")
best_val_f1 = 0.0
best_val_acc = 0.0
training_history = {
    'train_loss': [],
    'train_accuracy': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_f1': []
}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Training
    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
    print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_accuracy:.4f}")
    
    # Validation
    print("\nValidation Results:")
    val_preds, val_labels_actual, val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, device)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
    print("\nValidation Classification Report:")
    val_report = classification_report(val_labels_actual, val_preds, target_names=['Ham', 'Spam'])
    print(val_report)

    # Save validation classification report as PNG
    fig_val, ax_val = plt.subplots(figsize=(10, 6))
    ax_val.text(0.1, 0.5, val_report, fontsize=12, fontfamily='monospace', 
               verticalalignment='center', transform=ax_val.transAxes)
    ax_val.set_title('Validation Classification Report', fontsize=14, fontweight='bold')
    ax_val.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_classification_report_epoch_{epoch+1}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training history
    training_history['train_loss'].append(train_loss)
    training_history['train_accuracy'].append(train_accuracy)
    training_history['val_loss'].append(val_loss)
    training_history['val_accuracy'].append(val_accuracy)
    training_history['val_f1'].append(val_f1)
    
    # Save best model based on F1 score and accuracy
    if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_accuracy > best_val_acc):
        best_val_f1 = val_f1
        best_val_acc = val_accuracy
        print(f"New best validation F1: {best_val_f1:.4f}, Accuracy: {best_val_acc:.4f}. Saving model...")
        model.save_pretrained(f'{output_dir}/best_sms_model')
        tokenizer.save_pretrained(f'{output_dir}/best_sms_tokenizer')
        best_epoch = epoch + 1

# Save training history
with open(f'{output_dir}/training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)

# Load best model for final evaluation
print(f"\nLoading best model from epoch {best_epoch}...")
best_model = AutoModelForSequenceClassification.from_pretrained(f'{output_dir}/best_sms_model')
best_model.to(device)

# Final test evaluation with best model
print("\nFinal Test Results (Best Model):")
test_preds, test_labels_actual, test_loss, test_accuracy, test_f1 = evaluate(best_model, test_loader, device)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
print(f"Best model was from epoch {best_epoch}")
print("\nTest Classification Report:")
test_report = classification_report(test_labels_actual, test_preds, target_names=['Ham', 'Spam'])
print(test_report)

# Save test classification report as PNG (best model)
fig_test, ax_test = plt.subplots(figsize=(10, 6))
ax_test.text(0.1, 0.5, test_report, fontsize=12, fontfamily='monospace', 
            verticalalignment='center', transform=ax_test.transAxes)
ax_test.set_title(f'Test Classification Report (Best Model - Epoch {best_epoch})', 
                 fontsize=14, fontweight='bold')
ax_test.axis('off')
plt.tight_layout()
plt.savefig(f'{output_dir}/test_classification_report_best_model.png', 
           dpi=300, bbox_inches='tight')
plt.close()

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot training history
epochs = range(1, len(training_history['train_loss']) + 1)

# Loss plot
ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss')
ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(epochs, training_history['train_accuracy'], 'b-', label='Training Accuracy')
ax2.plot(epochs, training_history['val_accuracy'], 'r-', label='Validation Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

# Confusion matrix
cm = confusion_matrix(test_labels_actual, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
ax3.set_title('Confusion Matrix (Test Set)')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# Label distribution
label_counts = df['label'].value_counts()
ax4.bar(label_counts.index, label_counts.values, color=['skyblue', 'lightcoral'])
ax4.set_title('Dataset Label Distribution')
ax4.set_xlabel('Label')
ax4.set_ylabel('Count')
for i, v in enumerate(label_counts.values):
    ax4.text(i, v + 10, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{output_dir}/training_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Training completed!")
print(f"Best model saved to: {output_dir}/best_sms_model")
print(f"Training history saved to: {output_dir}/training_history.json")
print(f"Results visualization saved to: {output_dir}/training_results.png")
print(f"Best validation F1: {best_val_f1:.4f} (Epoch {best_epoch})")
print(f"Final test accuracy: {test_accuracy:.4f}")

# Test with 5 example messages
print("\n" + "="*60)
print("5 ÖRNEK MESAJ TESTİ")
print("="*60)

test_messages = [
    "Tebrikler! 1000 TL kazandınız! Hemen bu linke tıklayın: bit.ly/kazandin",
    "Merhaba, yarın saat 3'te toplantımız var. Unutma!",
    "ACIL! Hesabınız bloke edildi. Şifrenizi güncellemek için: fake-bank.com",
    "Anne, eve geç geleceğim. Akşam yemeğini bekleme.",
    "ÜCRETSİZ iPhone kazanmak için bu mesajı 10 kişiye gönder!"
]

label_names = ['HAM (Normal)', 'SPAM (İstenmeyen)']

for i, message in enumerate(test_messages, 1):
    # Preprocess the message
    clean_msg = preprocess_text(message)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        clean_msg,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    best_model.eval()
    with torch.no_grad():
        outputs = best_model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1).max().item()
    
    print(f"\nÖrnek {i}:")
    print(f"Mesaj: {message}")
    print(f"Tahmin: {label_names[prediction]}")
    print(f"Güven Skoru: {confidence:.4f} ({confidence*100:.2f}%)")
    print("-" * 50)