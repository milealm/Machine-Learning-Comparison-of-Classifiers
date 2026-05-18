# 1. Base image (lightweight)
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /Lab2

# 3. Copy dependency file first (for caching)
COPY requirements.txt .

# 4. Install dependencies 
# instala todas as bibliotecas listadas no arquivo
# não guarda cache dos pacotes (reduz o tamanho da imagem)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project files
# .. = Copia tudo da pasta atual para dentro do container
COPY . .

# 6. Run the app -> python knn.py. Para rodar varios, poe em um script e roda o script
CMD ["python", "svm.py"]