FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models.py client.py inference.py openenv.yaml openenv_state.py ./
COPY app.py mlops_environment.py artifact_generator.py ./
EXPOSE 7860
ENV PYTHONPATH=/app
USER appuser
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
