from fastapi import FastAPI, Body, status, HTTPException, UploadFile, File

from torch.nn.attention import SDPBackend, sdpa_kernel
from whisper import pipe


def create_app() -> FastAPI:
    api = FastAPI(
        title="Insanely Fast Whisper API",
        description='Whisper API',
        version="0.0.1",
        docs_url="/swagger",
        swagger_ui_parameters={"docExpansion": "none"},
        redoc_url="/redoc",
    )
    return api


app = create_app()


@app.get("",
         summary="Health Check",
         description="Health Check to verify that the application is running",
         status_code=status.HTTP_200_OK,
         response_description="The application is healthy",
         responses={
             status.HTTP_200_OK: {
                 "content": {
                     "application/json": {
                         "example": {"status": "healthy"}
                     }
                 }
             }
         })
async def health_check():
    """
    Health check endpoint to verify that the application is running.
    """
    return {"status": "healthy"}


@app.post("/",
          summary="Transcribe Audio",
          description="Uploads an audio file and transcribes or translates it using the Whisper model.",
          status_code=status.HTTP_200_OK,
          response_description="The transcribed or translated text along with timestamps.",
          responses={
              status.HTTP_200_OK: {
                  "content": {
                      "application/json": {
                          "example": {
                              "text": "Hello world!",
                              "chunks": [
                                  {"timestamp": [0.0, 1.5], "text": "Hello"},
                                  {"timestamp": [1.6, 3.0], "text": "world!"}
                              ]
                          }
                      }
                  }
              },
              status.HTTP_400_BAD_REQUEST: {
                  "description": "Invalid request parameters.",
                  "content": {
                      "application/json": {
                          "example": {"detail": "Invalid file format."}
                      }
                  }
              },
              status.HTTP_500_INTERNAL_SERVER_ERROR: {
                  "description": "Internal server error.",
                  "content": {
                      "application/json": {
                          "example": {"detail": "Error processing the audio file."}
                      }
                  }
              }
          })
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe."),
    batch_size: int = Body(
        default=64, description="Batch size for processing."),
    timestamp: str = Body(default="word", enum=[
                          "chunk", "word"], description="Timestamp granularity: word-level or chunk-level."),
):
    try:
        generate_kwargs = {
            "task": "transcribe",
            "language": "danish",
            "min_new_tokens": 256,
            "max_new_tokens": 256
        }

        file_content = await file.read()

        with sdpa_kernel(SDPBackend.MATH):
            outputs = pipe(
                file_content,
                chunk_length_s=30,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps="word" if timestamp == "word" else True,
            )

        return outputs

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
