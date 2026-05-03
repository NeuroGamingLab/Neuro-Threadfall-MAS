FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV THREADFALL_EMBED_DEVICE=cpu
ENV QDRANT_URL=http://qdrant:6333
ENV QDRANT_COLLECTION=threadfall_memories

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY threadfall ./threadfall
COPY .streamlit ./.streamlit

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

