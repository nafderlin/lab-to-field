# Lightweight image for the full Streamlit app (explanation pages + diagnosis): ONNX Runtime on CPU, no PyTorch.
#   docker build -t lab-to-field . && docker run -p 8501:8501 lab-to-field
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV PYTHONPATH=/home/user/app/src \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
