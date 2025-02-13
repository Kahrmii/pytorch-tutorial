MODEL_SAVE_PATH = "C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\Model\\qa_model.pth"
TRAINING_DATA_PATH = "C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\Data\\training_data"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import glob
import json
import random
from typing import List, Tuple, Dict
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(self, documents_dir: str, max_length: int = 512):
        self.documents = []
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        self.max_length = max_length
        
        # Lade alle .txt Dateien
        print("Lade Dokumente...")
        file_paths = glob.glob(os.path.join(documents_dir, "*.txt"))
        
        if not file_paths:
            raise ValueError(f"Keine .txt Dateien im Verzeichnis '{documents_dir}' gefunden. "
                           f"Bitte stellen Sie sicher, dass der Pfad korrekt ist und .txt Dateien enthält.")
        
        for file_path in tqdm(file_paths):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if not text:  # Überspringe leere Dateien
                        print(f"Warnung: Datei {file_path} ist leer und wird übersprungen.")
                        continue
                    # Teile Text in überlappende Abschnitte
                    chunks = self._create_chunks(text)
                    self.documents.extend(chunks)
            except Exception as e:
                print(f"Fehler beim Lesen der Datei {file_path}: {str(e)}")
                continue
        
        if not self.documents:
            raise ValueError("Keine gültigen Dokumente konnten geladen werden. "
                           "Bitte überprüfen Sie, ob die Textdateien Inhalt haben.")
    
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

class QuestionAnswering:
    def __init__(self, model_save_path: str = MODEL_SAVE_PATH, context_file: str = "C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\Data\\training_data\\context.txt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = QAModel().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        self.model_save_path = model_save_path
        self.context_file = context_file
        self.context = ""
        
        # Erstelle den Ordner für das Model, falls er nicht existiert
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Lade den Kontext, falls die Datei existiert
        self.load_context()
    
    def load_context(self):
        """Lädt den Kontext aus der Textdatei"""
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    self.context = f.read().strip()
                print(f"Kontext erfolgreich aus {self.context_file} geladen")
            else:
                print(f"Kontextdatei {self.context_file} nicht gefunden")
        except Exception as e:
            print(f"Fehler beim Laden des Kontexts: {str(e)}")
    
    def train(self, documents_dir: str, epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """Trainiert das Model mit dem geladenen Kontext"""
        if not self.context:
            raise ValueError("Kein Kontext geladen. Bitte stellen Sie sicher, dass die Kontextdatei existiert und nicht leer ist.")
        
        # Erstelle Dataset aus dem Kontext
        dataset = QADataset([self.context])  # Übergebe den Kontext als Liste
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
    
    def _find_relevant_context(self, question: str, window_size: int = 3) -> str:
        """Findet den relevantesten Kontext für eine Frage"""
        if not self.context:
            return ""
            
        # Teile den Kontext in Absätze
        paragraphs = [p.strip() for p in self.context.split('\n\n') if p.strip()]
        
        # Tokenisiere die Frage
        question_tokens = set(question.lower().split())
        
        # Bewerte jeden Absatz
        scored_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            # Berechne Worttreffer
            paragraph_tokens = set(paragraph.lower().split())
            word_score = len(question_tokens.intersection(paragraph_tokens))
            
            # Berechne Position (frühere Absätze leicht bevorzugt)
            position_score = 1 / (i + 1)
            
            # Kombiniere Scores
            total_score = word_score + 0.1 * position_score
            scored_paragraphs.append((total_score, paragraph))
        
        # Sortiere nach Score
        scored_paragraphs.sort(reverse=True)
        
        # Wähle die besten Absätze und deren Nachbarn
        selected_indices = set()
        for i, (score, _) in enumerate(scored_paragraphs[:3]):  # Top 3 Absätze
            if score > 0:  # Nur relevante Absätze
                for j in range(max(0, i-window_size), min(len(paragraphs), i+window_size+1)):
                    selected_indices.add(j)
        
        # Kombiniere ausgewählte Absätze
        selected_paragraphs = []
        for i in sorted(selected_indices):
            selected_paragraphs.append(paragraphs[i])
            
        return '\n\n'.join(selected_paragraphs)

    def answer_question(self, question: str) -> str:
        """Beantwortet eine Frage basierend auf dem relevanten Kontext"""
        if not self.context:
            return "Kein Kontext geladen. Bitte laden Sie zuerst einen Kontext."
            
        # Finde relevanten Kontext
        relevant_context = self._find_relevant_context(question)
        if not relevant_context:
            return "Kein relevanter Kontext gefunden."
            
        # Tokenisiere Frage und Kontext
        inputs = self.tokenizer(
            question,
            relevant_context,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generiere Antwort
        with torch.no_grad():
            start_logits, end_logits = self.model(input_ids, attention_mask)
            
            # Ignoriere spezielle Tokens ([CLS], [SEP], [PAD]) für Start und Ende
            sequence_len = torch.sum(attention_mask[0]).item()
            start_logits[0, sequence_len:] = float('-inf')
            end_logits[0, sequence_len:] = float('-inf')
            start_logits[0, 0] = float('-inf')  # [CLS]
            
            # Finde die besten Start- und End-Positionen
            start_scores = torch.softmax(start_logits, dim=1)
            end_scores = torch.softmax(end_logits, dim=1)
            
            # Stelle sicher, dass End nach Start kommt
            max_answer_len = 50  # Maximale Antwortlänge
            scores = torch.zeros((sequence_len, sequence_len))
            for start_idx in range(sequence_len):
                for end_idx in range(start_idx, min(start_idx + max_answer_len, sequence_len)):
                    scores[start_idx, end_idx] = start_scores[0, start_idx] * end_scores[0, end_idx]
            
            # Finde die besten Start- und End-Indizes
            scores = scores.cpu().numpy()
            start_idx, end_idx = np.unravel_index(np.argmax(scores), scores.shape)
            
            # Extrahiere und bereinige die Antwort
            answer_tokens = input_ids[0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Nachbearbeitung der Antwort
            answer = answer.strip()
            
            return answer if answer else "Keine passende Antwort gefunden."
        
    def add_document(self, text: str, category: str = "allgemein"):
        """Fügt ein neues Dokument zur Dokumentensammlung hinzu"""
        if category not in self.document_store:
            self.document_store[category] = []
        self.document_store[category].append(text)
        
    def load_documents_from_directory(self, directory: str):
        """Lädt Dokumente aus einem Verzeichnis mit Unterkategorien"""
        for root, dirs, files in os.walk(directory):
            category = os.path.basename(root) or "allgemein"
            for file in files:
                if file.endswith('.txt'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                self.add_document(text, category)
                    except Exception as e:
                        print(f"Fehler beim Lesen von {file}: {str(e)}")
    
    def find_relevant_context(self, question: str, top_k: int = 3) -> str:
        """Findet den relevantesten Kontext für eine Frage"""
        best_contexts = []
        question_tokens = set(question.lower().split())
        
        for category, documents in self.document_store.items():
            for doc in documents:
                # Teile Dokument in Absätze
                paragraphs = [p.strip() for p in doc.split('\n\n') if p.strip()]
                for paragraph in paragraphs:
                    # Berechne Relevanz-Score
                    paragraph_tokens = set(paragraph.lower().split())
                    score = len(question_tokens.intersection(paragraph_tokens))
                    if score > 0:
                        best_contexts.append((score, paragraph))
        
        # Sortiere nach Relevanz und kombiniere die besten Kontexte
        best_contexts.sort(reverse=True)
        combined_context = ' '.join(context for _, context in best_contexts[:top_k])
        
        return combined_context if combined_context else "Kein relevanter Kontext gefunden."
    
    def train(self, documents_dir: str, epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """Trainiert das Model auf den gegebenen Dokumenten"""
        # Lade zuerst alle Dokumente
        self.load_documents_from_directory(documents_dir)
        
        # Erstelle Dataset aus allen Dokumenten
        all_documents = []
        for documents in self.document_store.values():
            all_documents.extend(documents)
            
        dataset = QADataset(all_documents)
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
        
    def answer_question(self, question: str, context: str = None) -> str:
        """Beantwortet eine Frage basierend auf dem Kontext oder sucht relevanten Kontext"""
        if context is None:
            context = self.find_relevant_context(question)
        
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
        
        # Speichere das trainierte Model
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
            
            # Ignoriere spezielle Tokens ([CLS], [SEP], [PAD]) für Start und Ende
            sequence_len = torch.sum(attention_mask[0]).item()
            start_logits[0, sequence_len:] = float('-inf')
            end_logits[0, sequence_len:] = float('-inf')
            start_logits[0, 0] = float('-inf')  # [CLS]
            
            # Finde die besten Start- und End-Positionen
            start_scores = torch.softmax(start_logits, dim=1)
            end_scores = torch.softmax(end_logits, dim=1)
            
            # Stelle sicher, dass End nach Start kommt
            max_answer_len = 50  # Maximale Antwortlänge
            scores = torch.zeros((sequence_len, sequence_len))
            for start_idx in range(sequence_len):
                for end_idx in range(start_idx, min(start_idx + max_answer_len, sequence_len)):
                    scores[start_idx, end_idx] = start_scores[0, start_idx] * end_scores[0, end_idx]
            
            # Finde die besten Start- und End-Indizes
            scores = scores.cpu().numpy()
            start_idx, end_idx = np.unravel_index(np.argmax(scores), scores.shape)
            
            # Extrahiere und bereinige die Antwort
            answer_tokens = input_ids[0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Nachbearbeitung der Antwort
            answer = answer.strip()
            
            # Überprüfe ob die Antwort sinnvoll ist
            if not answer or len(answer.split()) < 2:
                best_sentence = self._find_best_sentence(context, question)
                return best_sentence
            
            return answer
            
    def _find_best_sentence(self, context: str, question: str) -> str:
        """Findet den relevantesten Satz im Kontext, falls keine gute Antwort gefunden wurde"""
        import re
        sentences = re.split('[.!?]', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "Keine passende Antwort gefunden."
            
        # Tokenisiere Frage und Sätze
        question_tokens = set(question.lower().split())
        
        best_score = 0
        best_sentence = sentences[0]
        
        for sentence in sentences:
            sentence_tokens = set(sentence.lower().split())
            # Berechne Überlappung zwischen Frage und Satz
            score = len(question_tokens.intersection(sentence_tokens))
            if score > best_score:
                best_score = score
                best_sentence = sentence
                
        return best_sentence.strip()

# Beispiel zur Verwendung:
if __name__ == "__main__":
    # Initialisiere das QA-System
    qa = QuestionAnswering(model_save_path=MODEL_SAVE_PATH)
    
    # Training
    print(f"Starte Training mit Daten aus: {TRAINING_DATA_PATH}")
    print(f"Model wird gespeichert unter: {MODEL_SAVE_PATH}")
    
    # qa.train(TRAINING_DATA_PATH, epochs=3)
    
    # Oder lade ein bereits trainiertes Model
    qa.load_model()
    
    # Beispiel für eine Frage
    context = qa.find_relevant_context("Wofür wird PyTorch verwendet?")
    frage = "Wofür wird PyTorch verwendet?"
    
    antwort = qa.answer_question(frage, context)
    print(f"\nFrage: {frage}")
    print(f"Antwort: {antwort}")