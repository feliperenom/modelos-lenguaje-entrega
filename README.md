## ¿Qué hace este codigo? 

Este es un asistente virtual en formato de chatbot que responde consultas del usuario sobre las siguientes leyes, basado en el programa de lenguaje ciudadano de IMPO: 

* Ley de Inclusión financiera
* Ley de Residuos
* Ley de Defensa del consumidor
* Ley de Licencias Especiales

## Dockerización de la aplicación 

Para poder construir los contenedores de esta aplicación, deberás tener Docker instalado en tu computadora. 

Luego que tienes Docker instalado, abre tu editor de código de preferencia y abre la carpeta del proyecto. 

Una vez ahi, primero vamos a dockerizar el backend. Para eso correrás los siguientes comandos: 

1. `cd [ruta de dónde tienes guardado el proyecto + src/model] docker build -f back.Dockerfile -t backend-legaluy .`

2. `docker run -p 8080:8080 backend-legaluy`

Luego, vamos a dockerizar el frontend, para ello correras los siguientes comandos: 

1. `cd [ruta de dónde tienes guardado el proyecto + src/front] docker build -f front.Dockerfile -t frontend-legaluy .`
2. `docker run -p 8501:8501 frontend-legaluy`
