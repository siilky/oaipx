FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user proxy.py proxy.py
COPY --chown=user index.html index.html
CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "7860"]