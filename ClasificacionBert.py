# Importamos las bibliotecas necesarias
import torch  # Para trabajar con tensores y la inferencia en el modelo
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments  # Hugging Face para modelos NLP
from datasets import Dataset, DatasetDict  # Para manejar datasets estructurados
import numpy as np  # Para trabajar con cálculos numéricos
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # Métricas para evaluar el modelo

# 📌 Paso 1: Definir los datos de entrenamiento
texts = [
    "Los paneles solares son una fuente eficiente de energía renovable.",
    "El carbón sigue siendo una fuente importante de energía en muchos países.",
    "La energía eólica está ganando popularidad en todo el mundo.",
    "Las centrales nucleares son controversiales pero producen energía sin emisiones de CO2.",
    "La biomasa es una forma de energía renovable que utiliza materiales orgánicos.",
]
labels = [1, 0, 1, 0, 1]  # 1 para energía renovable, 0 para energía no renovable

# 📌 Paso 2: Crear un Dataset de Hugging Face
# Convierte los datos en un objeto DatasetDict para poder manipularlo fácilmente en el entrenamiento
dataset = DatasetDict({"train": Dataset.from_dict({"text": texts, "label": labels})})

# 📌 Paso 3: Cargar el tokenizador y el modelo pre-entrenado BERT
# "bert-base-uncased" es un modelo de BERT en inglés sin distinción entre mayúsculas y minúsculas
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # num_labels=2 indica clasificación binaria

# 📌 Paso 4: Función para convertir los textos en tokens
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Aplicamos la función a nuestro dataset para convertir los textos en vectores numéricos
tokenized_dataset = dataset.map(tokenize_function)

# 📌 Paso 5: Dividir los datos en entrenamiento (80%) y prueba (20%)
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)

# Convertimos el dataset dividido en un DatasetDict para usarlo en entrenamiento
dataset = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

# 📌 Paso 6: Definir una función para calcular métricas de rendimiento
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids  # Extraemos predicciones y etiquetas reales
    predictions = np.argmax(logits, axis=-1)  # Convertimos los logits en clases (0 o 1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 📌 Paso 7: Configurar los parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",  # Carpeta donde se guardarán los resultados
    num_train_epochs=3,  # Número de veces que el modelo verá los datos de entrenamiento
    per_device_train_batch_size=8,  # Tamaño del batch en entrenamiento
    per_device_eval_batch_size=8,  # Tamaño del batch en evaluación
    warmup_steps=500,  # Pasos de calentamiento para optimización
    weight_decay=0.01,  # Regularización para evitar sobreajuste
    logging_dir='./logs',  # Carpeta donde se guardarán los logs de entrenamiento
    logging_steps=10,  # Se registrarán logs cada 10 pasos
    evaluation_strategy="epoch",  # Se evaluará el modelo al final de cada época
)

# 📌 Paso 8: Crear el Trainer (gestiona el entrenamiento y evaluación del modelo)
trainer = Trainer(
    model=model,  # Modelo BERT para clasificación
    args=training_args,  # Parámetros de entrenamiento
    train_dataset=dataset["train"],  # Datos de entrenamiento
    eval_dataset=dataset["test"],  # Datos de evaluación
    compute_metrics=compute_metrics,  # Función de métricas para evaluación
)

# 📌 Paso 9: Entrenar el modelo
trainer.train()

# 📌 Paso 10: Evaluar el modelo después del entrenamiento
eval_results = trainer.evaluate()
print(eval_results)  # Muestra las métricas de rendimiento del modelo

# 📌 Paso 11: Hacer una predicción con un nuevo texto
text = "La energía geotérmica aprovecha el calor de la Tierra."
inputs = tokenizer(text, return_tensors="pt")  # Convertimos el texto en tensores de entrada para PyTorch

# Desactivamos el cálculo de gradientes ya que solo queremos hacer inferencia
with torch.no_grad():
    outputs = model(**inputs)  # Pasamos los datos al modelo y obtenemos la salida

# Extraemos la predicción y la convertimos en una categoría (0 o 1)
prediction = torch.argmax(outputs.logits, dim=-1).item()

# Mostramos la predicción final
print(f"Predicción: {'Energía Renovable' if prediction == 1 else 'Energía No Renovable'}")
