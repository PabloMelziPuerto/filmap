import pandas as pd
import nltk
import string
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Descargar recursos necesarios de NLTK
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

# Cargar los datos
from nltk.corpus import movie_reviews

# Cargo reseñas
positive_reviews = movie_reviews.fileids('pos')
negative_reviews = movie_reviews.fileids('neg')

# Extraer los datos y las etiquetas
reviews_data = []
labels = []

for fileid in positive_reviews:
    reviews_data.append(movie_reviews.raw(fileid))
    labels.append(1)  # Sentimiento positivo (1)

for fileid in negative_reviews:
    reviews_data.append(movie_reviews.raw(fileid))
    labels.append(0)  # Sentimiento negativo (0)


# Eliminar puntuación y stopwords, y aplicar stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar puntuación
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenizar el texto
    tokens = nltk.word_tokenize(text)
    
    # Eliminar stopwords y aplicar stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Preprocesar todas las reseñas
preprocessed_reviews = [preprocess_text(review) for review in reviews_data]

# Vectorización usando TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(preprocessed_reviews)
y = labels

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento del modelo (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)

# Evaluar precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy:.4f}')

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.title("Matriz de Confusión")
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Probar el modelo con algunas entradas personalizadas
def predict_sentiment(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return "Positivo" if prediction == 1 else "Negativo"

#Ejemplo uso
test_text = "I love this movie! It was amazing!"
print(f"Sentimiento de '{test_text}': {predict_sentiment(test_text)}")

test_text = "This movie was terrible and boring."
print(f"Sentimiento de '{test_text}': {predict_sentiment(test_text)}")
