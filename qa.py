import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import glob
import json
import random
from typing import List, Tuple, Dict
from tqdm import tqdm



# Setup
training_data = 'C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\Data'



class QADataset(Dataset):
    def __init__(self, documents_dir: str, max_length: int = 512):
        self.documents = []
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        self.max_length = max_length
        
        # Lade alle .txt Dateien
        print("Lade Dokumente...")
        file_paths = glob.glob(os.path.join(documents_dir, "*.txt"))
        for file_path in tqdm(file_paths):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Teile Text in überlappende Abschnitte
                chunks = self._create_chunks(text)
                self.documents.extend(chunks)
    
    def _create_chunks(self, text: str, overlap: int = 128) -> List[str]:
        """Erstellt überlappende Textabschnitte"""
        words = text.split()
        chunk_size = self.max_length - 2  # Berücksichtige [CLS] und [SEP] tokens
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.documents[idx]
        
        # Erstelle ein "synthetisches" Frage-Antwort-Paar
        answer_start = random.randint(0, max(0, len(text.split()) - 10))
        answer_words = text.split()[answer_start:answer_start + 10]
        answer = ' '.join(answer_words)
        
        # Erstelle eine einfache Frage (kann später verbessert werden)
        question = f"Was steht im Text über: {' '.join(answer_words[:3])}?"
        
        # Tokenisiere Frage und Kontext
        encoding = self.tokenizer(
            question,
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Finde Start- und End-Position der Antwort
        answer_encoding = self.tokenizer(answer, add_special_tokens=False)
        answer_start_token = None
        answer_end_token = None
        
        for i in range(len(encoding['input_ids'][0])):
            if encoding['input_ids'][0][i:i+len(answer_encoding['input_ids'])].tolist() == answer_encoding['input_ids']:
                answer_start_token = i
                answer_end_token = i + len(answer_encoding['input_ids']) - 1
                break
        
        # Falls keine exakte Position gefunden wurde, verwende Dummy-Positionen
        if answer_start_token is None:
            answer_start_token = 0
            answer_end_token = 1
            
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(answer_start_token),
            'end_positions': torch.tensor(answer_end_token)
        }



# Model
class QAModel(nn.Module):
    def __init__(self, pretrained_model: str = 'bert-base-german-cased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  # 2 für start/end logits
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)



# Question Answering System
class QuestionAnswering:
    def __init__(self, model_save_path: str = 'C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\Model\\qa_model.pth'):
        self.device = torch.device('cpu') # weil kein Cuda oder ROCm verfügbar (auch kein DirectML weil FK MS)
        self.model = QAModel().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        self.model_save_path = model_save_path
        
    def train(self, documents_dir: str, epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """Trainiert das Model auf den gegebenen Dokumenten"""
        dataset = QADataset(documents_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                start_logits, end_logits = self.model(input_ids, attention_mask)
                
                # Berechne Verlust für Start- und End-Positionen
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Durchschnittlicher Loss: {epoch_loss / len(dataloader):.4f}")
        
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model gespeichert unter {self.model_save_path}")
    
    def load_model(self):
        """Lädt ein gespeichertes Model"""
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
    
    def answer_question(self, question: str, context: str) -> str:
        """Beantwortet eine Frage basierend auf dem gegebenen Kontext"""
        self.model.eval()
        
        # Tokenisiere Eingabe
        inputs = self.tokenizer(
            question,
            context,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            start_logits, end_logits = self.model(input_ids, attention_mask)
            
            # Finde die wahrscheinlichsten Start- und End-Positionen
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
            
            # Stelle sicher, dass End-Position nach Start-Position kommt
            if end_idx < start_idx:
                end_idx = start_idx + 1
            
            # Decodiere die Antwort
            answer_tokens = input_ids[0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens)
            
            return answer



# Beispiel zur Verwendung:
if __name__ == "__main__":
    # Initialisiere das QA-System
    qa = QuestionAnswering()
    
    # Training
    qa.train(training_data, epochs=3)
    
    # Oder lade ein bereits trainiertes Model
    # qa.load_model()
    
    # Beispiel für eine Frage
    context = "PyTorch ist eine Open-Source-Machine-Learning-Bibliothek. Sie wird hauptsächlich für Deep Learning verwendet."
    frage = "Wofür wird PyTorch verwendet?"
    
    antwort = qa.answer_question(frage, context)
    print(f"Frage: {frage}")
    print(f"Antwort: {antwort}")