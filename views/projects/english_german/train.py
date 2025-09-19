import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_loader import create_dataloaders
from model import TranslationModel
from tqdm import tqdm
import time
import numpy as np

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, tokenizer, device):
    model.eval()
    total_loss = 0
    
    # Test sentences for monitoring progress
    test_sentences = [
        "Hello, how are you?",
        "I love learning German.",
        "The weather is nice today."
    ]
    
    with torch.no_grad():
        # Calculate validation loss
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
        
        # Translate test sentences
        print("\nTest Translations:")
        for sentence in test_sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            outputs = model.generate(input_ids, attention_mask)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"EN: {sentence}")
            print(f"DE: {translation}\n")
    
    return total_loss / len(dataloader)

def train():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, tokenizer = create_dataloaders(batch_size=4)
    
    # Initialize model
    model = TranslationModel().to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    # Training parameters
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Training loop
    print("Starting training...")
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device)
            
            # Validate
            val_loss = validate(model, val_loader, tokenizer, device)
            
            # Update learning rate
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print("Saving best model...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model_en_de.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
    finally:
        print("\nTraining completed!")

if __name__ == "__main__":
    train()