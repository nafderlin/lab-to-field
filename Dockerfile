# Lightweight image for the Streamlit demo: ONNX Runtime on CPU, no PyTorch.
# Used by the Hugging Face Space (see deploy/space/) and works anywhere Docker runs:
#   docker build -t lab-to-field . && docker run -p 8501:8501 lab-to-field
FROM python:3.12-slim

# Hugging Face Spaces run containers as user 1000.
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

# XSRF protection blocks photo uploads inside the Hugging Face iframe; the app stores no user data.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
