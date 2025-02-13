import os
import json

def create_training_structure():
    # Basis-Verzeichnisstruktur
    categories = {
        'informatik': ['programmierung', 'datenbanken', 'algorithmen'],
        'wirtschaft': ['marketing', 'finanzen', 'management'],
        'medizin': ['anatomie', 'krankheiten', 'behandlungen']
    }
    
    # Erstelle Hauptverzeichnis
    os.makedirs('C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\Data\\training_data', exist_ok=True)
    
    # Erstelle Kategorien und Beispieldateien
    for category, topics in categories.items():
        category_path = os.path.join('C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\Data\\training_data', category)
        os.makedirs(category_path, exist_ok=True)
        
        for topic in topics:
            file_path = os.path.join(category_path, f"{topic}.txt")
            create_example_content(file_path, category, topic)

def create_example_content(file_path, category, topic):
    """Erstellt strukturierten Inhalt für eine Trainingsdatei"""
    content = f"""THEMA: {topic.title()}
KONTEXT: Grundlagen von {topic.title()}

Dies ist ein Beispieltext über {topic}. 
Hier kommen die wichtigsten Grundlagen und Konzepte.
Der Text sollte mehrere Absätze enthalten und ausführlich sein.

THEMA: Fortgeschrittene {topic.title()}
KONTEXT: Vertiefung {topic.title()}

Dieser Abschnitt behandelt fortgeschrittene Konzepte von {topic}.
Wichtige Aspekte sind:
- Konzept 1
- Konzept 2
- Konzept 3

THEMA: Praktische Anwendung
KONTEXT: Anwendungsbeispiele

Hier folgen praktische Beispiele und Anwendungsfälle.
Diese helfen beim Verständnis der Konzepte."""

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def add_training_content(category: str, topic: str, content: str):
    """Fügt neuen Trainingsinhalt hinzu"""
    file_path = os.path.join('training_data', category, f"{topic}.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\nTHEMA: {topic}\nKONTEXT: Neuer Inhalt\n\n{content}")

# Beispiel zur Verwendung
if __name__ == "__main__":
    # Erstelle Grundstruktur
    create_training_structure()
    
    # Füge neuen Inhalt hinzu
    neuer_inhalt = """
    Python ist eine vielseitige Programmiersprache.
    Sie wird häufig in folgenden Bereichen eingesetzt:
    - Webentwicklung
    - Datenanalyse
    - Künstliche Intelligenz
    """
    add_training_content('informatik', 'programmierung', neuer_inhalt)