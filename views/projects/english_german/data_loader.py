import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianTokenizer
from typing import List, Tuple

class TranslationDataset(Dataset):
    def __init__(self, english_texts: List[str], german_texts: List[str], tokenizer, max_length: int = 128):
        self.english_texts = english_texts
        self.german_texts = german_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.english_texts)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        eng_text = self.english_texts[idx]
        ger_text = self.german_texts[idx]

        # Tokenize the texts
        eng_encoded = self.tokenizer(eng_text, 
                                   max_length=self.max_length, 
                                   padding='max_length', 
                                   truncation=True, 
                                   return_tensors='pt')
        
        ger_encoded = self.tokenizer(ger_text, 
                                   max_length=self.max_length, 
                                   padding='max_length', 
                                   truncation=True, 
                                   return_tensors='pt')

        return {
            'input_ids': eng_encoded['input_ids'].squeeze(),
            'attention_mask': eng_encoded['attention_mask'].squeeze(),
            'labels': ger_encoded['input_ids'].squeeze()
        }

def get_train_data() -> Tuple[List[str], List[str]]:
    """Returns sample training data"""
    english_texts = [
        "Hello, how are you?",
        "I love learning German.",
        "The weather is nice today.",
        "What is your name?",
        "I live in Berlin.",
        "Do you speak English?",
        "The food is delicious.",
        "I need help.",
        "See you tomorrow.",
        "Have a nice day!"
    ]
    
    german_texts = [
        "Hallo, wie geht es dir?",
        "Ich liebe es, Deutsch zu lernen.",
        "Das Wetter ist heute schön.",
        "Wie heißt du?",
        "Ich wohne in Berlin.",
        "Sprechen Sie Englisch?",
        "Das Essen ist köstlich.",
        "Ich brauche Hilfe.",
        "Bis morgen.",
        "Einen schönen Tag!"
    ]
    
    return english_texts, german_texts

def create_dataloaders(batch_size: int = 4) -> Tuple[DataLoader, DataLoader, MarianTokenizer]:
    """Creates training and validation dataloaders"""
    
    # Load tokenizer
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    
    # Get data
    english_texts, german_texts = get_train_data()
    
    # Split into train/val
    train_size = int(0.8 * len(english_texts))
    
    train_eng = english_texts[:train_size]
    train_ger = german_texts[:train_size]
    val_eng = english_texts[train_size:]
    val_ger = german_texts[train_size:]
    
    # Create datasets
    train_dataset = TranslationDataset(train_eng, train_ger, tokenizer)
    val_dataset = TranslationDataset(val_eng, val_ger, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    # Test the dataloader
    train_loader, val_loader, tokenizer = create_dataloaders()
    
    for batch in train_loader:
        print("Input shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        print("Labels shape:", batch['labels'].shape)
        break