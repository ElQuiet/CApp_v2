# Contact Angle Analyzer — container untuk Hugging Face Spaces (atau server mana pun yang menjalankan Docker).
# Versi Python & library sama dengan yang sudah dites (requirements.txt).
FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces menjalankan container sebagai user dengan UID 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
USER user

EXPOSE 7860
# enableXsrfProtection=false: Spaces menampilkan app di dalam iframe; tanpa ini upload foto ditolak (403).
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
