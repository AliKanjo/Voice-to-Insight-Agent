# Voice-to-Insight Agent: container image for Hugging Face Spaces (Docker SDK).
# Also runs anywhere Docker does (Render, Railway, Fly.io, a VPS).
#
#   docker build -t voice-to-insight .
#   docker run -p 7860:7860 -e GEMINI_API_KEY=... voice-to-insight
#
# Defaults below are for the PUBLIC demo (sign-in required, daily caps on your key).
# For a private instance, set DEMO_MODE=false and APP_PASSWORD at runtime.

FROM python:3.13-slim

# Spaces run the container as UID 1000; create that user and work in its home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR $HOME/app

# Dependencies first, so code changes don't reinstall them.
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# Bake the Whisper model into the image so the first visitor doesn't wait for a download.
# Change with: docker build --build-arg WHISPER_MODEL=medium (and set the same at runtime).
ARG WHISPER_MODEL=small
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

COPY --chown=user . .

# Runtime defaults. Hugging Face Space "Variables and secrets" override any of these.
# Put GEMINI_API_KEY in the Space's Secrets, never here.
ENV PORT=7860 \
    WHISPER_MODEL=${WHISPER_MODEL} \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    TRANSCRIBE_BACKEND=local \
    UNDERSTAND_BACKEND=gemini \
    GEMINI_FALLBACK_MODEL=gemini-3.8-flash \
    DEMO_MODE=true \
    MAX_UPLOAD_MB=10 \
    DB_PATH=/home/user/app/data/insights.db

EXPOSE 7860

# --proxy-headers: Spaces sit behind a proxy, so per-IP limits must read the real client IP.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
