# Usa una imagen oficial de Python como base
FROM python:3.11-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos de requerimientos
COPY back-requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r back-requirements.txt

# Copia el resto del código fuente al contenedor
COPY . .

# Expone el puerto en el que correrá FastAPI (por defecto 8080)
EXPOSE 8080

# Comando para iniciar el servidor FastAPI con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]