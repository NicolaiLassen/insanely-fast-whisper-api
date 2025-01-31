# Whisper Large v3 Turbo Server

uvicorn main:app --workers 1 --host 0.0.0.0 --port 8080 --timeout-keep-alive 300