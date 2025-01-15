# Das Modell ist ein einfaches Feedforward-Neuronales Netzwerk, das darauf trainiert wird, handgeschriebene Ziffern aus dem MNIST-Datensatz zu erkennen. Der MNIST-Datensatz besteht aus Bildern von handgeschriebenen Ziffern (0-9), und das Ziel des Modells ist es, diese Ziffern korrekt zu klassifizieren.

# Ablauf:
# Eingabe verarbeiten: Das Modell nimmt ein Bild mit 28x28 Pixeln als Eingabe, was insgesamt 784 Pixel ergibt.
# Verarbeitung durch eine versteckte Schicht: Die Eingabedaten werden durch eine versteckte Schicht mit 500 Neuronen verarbeitet.
# Ausgabe erzeugen: Das Modell gibt eine Vorhersage für eine von 10 Klassen (Ziffern 0-9) aus.
# Das Modell wird mit dem MNIST-Datensatz trainiert, um die Gewichte und Biases so anzupassen, dass es die Ziffern möglichst genau erkennt. Nach dem Training kann das Modell verwendet werden, um neue, unbekannte handgeschriebene Ziffern zu klassifizieren.


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
device = torch.device('cpu') # nur CPU da wir Cuda nicht verwenden können


# Zunächst definieren wir einige Hyperparameter für unser Modell. Diese Parameter steuern das Verhalten des Modells und die Trainingsphase.
input_size = 784                                                        # 28x28=784 Pixels / anzahl der Eingabeneuronen (eins für jedes Pixel)
hidden_size = 500                                                       # Anzahl der Neuronen in der Hidden Layer
num_classes = 10                                                        # 10 Klassen (eine Klasse für jedes Zeichen)
num_epochs = 5                                                          # Anzahl der Epochen (mehr Epochen = bessere Genauigkeit aber auch längere Trainingszeit)
batch_size = 100                                                        # Anzahl der Trainingsbeispiele pro Batch
learning_rate = 0.001                                                   # Lernrate (wie schnell das Modell lernt und wie schnell es sich anpasst)


# um das Modell zu trainieren, benötigen wir einen Datensatz. In diesem Fall verwenden wir den MNIST-Datensatz, der in PyTorch verfügbar ist.
# MNIST ist ein Datensatz, der aus 60.000 Trainingsbildern und 10.000 Testbildern besteht. Jedes Bild ist ein 28x28 Pixel Bild, das ein handgeschriebenes Ziffern von 0 bis 9 darstellt.
# falls der Datensatz nicht vorhanden ist, wird er heruntergeladen (siehe download=True)
train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=True) # erstellen des Trainingsdatensatzes
test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor()) # erstellen des Testdatensatzes


# PyTorch bietet DataLoader-Klassen, die das Laden von Daten in Batches vereinfachen. Diese Klassen ermöglichen es uns, die Daten in Batches zu laden und sie an das Modell weiterzugeben.	
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # laden des Trainingsdatensatzes
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # laden des Testdatensatzes


# Nun werden wir die Architektur unseres Modells definieren. In diesem Fall verwenden wir ein einfaches Feedforward-Neuronales Netzwerk mit nur einer versteckten Schicht.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):           # Initialisierung des Modells und der Schichten
        super(NeuralNet, self).__init__()                               # Initialisierung der Basisklasse nn.Module (stellt sicher, dass die Klasse NeuralNet ordnungsgemäß initialisiert wird und alle grundlegenden Eigenschaften und Methoden von nn.Module zur Verfügung stehen)
        self.fc1 = nn.Linear(input_size, hidden_size)                   # erstellen des input Layers (Anwendung lineare Transformation auf die Eingabedaten)
        self.relu = nn.ReLU()                                           # erstellen der ReLU Aktivierungsfunktion (Anwendung nichtlineare Transformation auf die Ausgaben der ersten Schicht)
        self.fc2 = nn.Linear(hidden_size, num_classes)                  # erstellen des output Layers (Diese Schicht transformiert die Ausgaben der versteckten Schicht in die Ausgaben des Netzwerks)
    
    def forward(self, x):                                               # Vorwärtsdurchlauf des Modells (x = Eingabetensor)
        out = self.fc1(x)                                               # Anwendung der ersten Schicht. x wird durch die erste voll verbundene Schicht self.fc1 geleitet. Das Ergebnis dieser Transformation wird in der Variablen out gespeichert.
        out = self.relu(out)                                            # Anwendung der ReLU Aktivierungsfunktion auf die Ausgabe der ersten Schicht. Die Ausgabe wird in der Variablen out gespeichert.
        out = self.fc2(out)                                             # Anwendung der zweiten Schicht. Die Ausgabe der ReLU-Aktivierungsfunktion wird durch die zweite voll verbundene Schicht self.fc2 geleitet. Das Ergebnis dieser Transformation wird erneut in der Variablen out gespeichert.
        return out                                                      # Rückgabe der Ausgabe des Modells   
model = NeuralNet(input_size, hidden_size, num_classes).to(device)      # nun erstellen wir eine Instanz unseres Modells und übergeben die Hyperparameter input_size, hidden_size und num_classes. Anschließend verschieben wir das Modell auf das vorher festgelegte gerät mit der Methode to(device).


# Anschließend definieren wir den Verlust und den Optimierer für unser Modell.
criterion = nn.CrossEntropyLoss()                                       # Der Verlust ist die Funktion, die wir minimieren möchten, um das Modell zu trainieren. In diesem Fall verwenden wir die Kreuzentropieverlustfunktion, die häufig für Klassifikationsprobleme verwendet wird.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)      # Der Optimierer ist der Algorithmus, der die Gewichte des Modells anpasst, um den Verlust zu minimieren. In diesem Fall verwenden wir den Adam-Optimierer, der eine effiziente Methode zur Anpassung der Gewichte des Modells bietet.


# Jetzt werden wir das Modell trainieren. Der Trainingsprozess besteht aus mehreren Schritten, die in verschachtelten Schleifen ausgeführt werden.
# Die äußere Schleife iteriert über die Epochen, während die innere Schleife über die Batches der Trainingsdaten iteriert.
# Im Grunde genommen bedeuted das, dass nach jedem Durchlauf die Gewichte des Modells angepasst werden, um den Verlust zu minimieren und das Model zu verfeinern.
total_step = len(train_loader)                                          # Berechnung der Gesamtanzahl der Batches im Trainingsdatensatz  
for epoch in range(num_epochs):                                         # Epochen sind Iterationen über den gesamten Trainingsdatensatz (jede Epoche besteht aus mehreren Batches)
    for i, (images, labels) in enumerate(train_loader):  
        images = images.reshape(-1, 28*28).to(device)                   # Verschieben der Eingabedaten und der Labels auf das Gerät und Ändern der Form der Eingabedaten
        labels = labels.to(device)  	                                # Verschieben der Eingabedaten und der Labels auf das Gerät und Ändern der Form der Eingabedaten
        
        outputs = model(images)                                         # Vorwärtsdurchlauf des Modells, um die Ausgaben zu berechnen (Vorhersagen des Modells)
        loss = criterion(outputs, labels)                               # Berechnung des Verlusts, indem die Ausgaben des Modells und die Labels verglichen werden
        
        optimizer.zero_grad()                                           # Zurücksetzen der Gradienten auf Null, um sicherzustellen, dass die Gradienten nicht akkumuliert werden (sich anhäufen und zu unerwartetem Verhalten führen)
        loss.backward()                                                 # Rückwärtsdurchlauf des Modells, um die Gradienten der Verlustfunktion zu berechnen
        optimizer.step()                                                # Nachdem die Gradienten der Verlustfunktion in Bezug auf die Modellparameter durch den Rückwärtsdurchlauf (loss.backward()) berechnet wurden, verwendet optimizer.step() diese Gradienten, um die Parameter des Modells zu aktualisieren.
        
        if (i+1) % 100 == 0:                                            # Ausgabe der Verlustfunktion alle 100 Schritte
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            

# Nachdem das Modell trainiert wurde, können wir es auf dem Testdatensatz evaluieren, um die Genauigkeit des Modells zu überprüfen.  
with torch.no_grad():                                                   # Der gesamte Block ist in torch.no_grad() eingeschlossen, was bedeutet, dass während der Ausführung keine Gradienten berechnet werden um Speicherplatz zu sparen und Laufzeit zu optimieren.
    correct = 0
    total = 0                                                           # Initialisierung der Variablen correct und total auf 0	um fortlaufend die Anzahl der korrekten Vorhersagen und die Gesamtanzahl der Testbeispiele zu zählen
    for images, labels in test_loader:                                  # Iteration über alle Batches des Testdatensatzes
        images = images.reshape(-1, 28*28).to(device)                   # Verschieben der Eingabedaten und der Labels auf das Gerät und Ändernung der Form der Eingabedaten
        labels = labels.to(device)                                      
        outputs = model(images)                                         # Vorwärtsdurchlauf des Modells, um die Ausgaben zu berechnen (Vorhersagen des Modells)
        _, predicted = torch.max(outputs.data, 1)                       # Berechnung der Vorhersagen des Modells, indem der Index des maximalen Werts in den Ausgaben des Modells verwendet wird
        total += labels.size(0)                                         # Erhöhung der Gesamtanzahl der Testbeispiele um die Anzahl der Labels im aktuellen Batch
        correct += (predicted == labels).sum().item()                   # Erhöhung der Anzahl der korrekten Vorhersagen um die Anzahl der korrekten Vorhersagen im aktuellen Batch

    print('Accuracy of the network on the 10000 test images: {} %'      # Ausgabe der Genauigkeit des Modells auf dem Testdatensatz
          .format(100 * correct / total))
    

# Nun haben wir unser Modell erfolgreich trainiert und evaluiert. Um das Modell für zukünftige Verwendungszwecke zu speichern, können wir es als Checkpoint speichern.
torch.save(model.state_dict(), 'model.ckpt')