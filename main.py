from flask import Flask, request, jsonify
import whisper
import werkzeug
import os

app = Flask(__name__)

# Load the Whisper model
model = whisper.load_model("tiny")

@app.route('/whisper', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Optional language parameter from form-data
    language = request.form.get('language', None)

    if file and werkzeug.utils.secure_filename(file.filename):
        # Save the uploaded audio file
        filepath = werkzeug.utils.secure_filename(file.filename)
        file.save(filepath)

        # Load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(filepath)
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Specify language in DecodingOptions if provided
        if language:
            options = whisper.DecodingOptions(language=language, fp16=False)
            print(f"Transcribing in specified language: {language}")
        else:
            # Detect the spoken language if not specified
            _, probs = model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            options = whisper.DecodingOptions(fp16=False)
            print(f"Detected language: {detected_language}")

        result = model.decode(mel, options)

        # Clean up the audio file
        os.remove(filepath)

        return jsonify({"text": result.text, "language": language if language else detected_language})

    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    from waitress import serve
    # Get host and port from environment variables, use default values if not provided
    host = os.environ.get('WHISPER_HOST', '0.0.0.0')
    port = int(os.environ.get('WHISPER_PORT', 28466))
    app.run(host=host, port=port)