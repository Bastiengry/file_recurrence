# Utilise une image Python officielle légère
FROM python:3.11-slim

# Installation des dépendances système pour le rendu graphique
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier UNIQUEMENT le fichier des dépendances
COPY requirements.txt .

# Installer les bibliothèques (étape longue)
RUN pip install --no-cache-dir -r requirements.txt

# Déclaration des volumes (facultatif mais recommandé pour la doc)
VOLUME ["/app/data", "/app/output"]

COPY . .

CMD ["python", "main.py"]