from transformers import MarianMTModel
import torch
import torch.nn as nn
from typing import Dict, Union

class TranslationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                labels: torch.Tensor = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs (optional)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate translations
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            max_new_tokens=128
        )