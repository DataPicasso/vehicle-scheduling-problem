# Usa Python 3.9 como base
FROM python:3.9

# Configura el directorio de trabajo
WORKDIR /app

# Copia todos los archivos del proyecto al contenedor
COPY . /app

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto 8501 (por defecto en Streamlit)
EXPOSE 8501

# Comando para ejecutar la app en Cloud Run
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
