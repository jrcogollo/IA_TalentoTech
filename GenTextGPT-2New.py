from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo y el tokenizador de GPT-2 desde Hugging Face
model_name = "gpt2"  # También puedes usar "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Texto de entrada (prompt)
input_text = "Once upon a time in a distant galaxy, a spaceship"

# Tokenizar el texto de entrada
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generar texto con el modelo GPT-2
output = model.generate(
    input_ids,
    max_length=100,  # Número máximo de tokens generados
    num_return_sequences=1,  # Número de secuencias a generar
    temperature=0.7,  # Controla la aleatoriedad (0.7 es moderado)
    top_k=50,  # Considera solo los 50 tokens más probables en cada paso
    top_p=0.9,  # Nucleus sampling: mantiene solo el top 90% de probabilidad acumulada
)

# Decodificar y mostrar la salida
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
