# Usa una imagen oficial de Python como base
FROM python:3.11-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos de requerimientos
COPY requirements-front.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements-front.txt

# Copia el resto del código fuente al contenedor
COPY . .

# Expone el puerto por defecto de Streamlit
EXPOSE 8501

# Comando para iniciar la app de Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]