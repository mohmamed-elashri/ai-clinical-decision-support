"""Single-process launcher for the Hugging Face Docker Space.

The public listener is Gradio on port 7860. Flask remains the internal API
on port 5000, preserving the existing frontend/backend contract.
"""

import logging
import os
from threading import Thread

from waitress import serve as serve_flask

from app import app
from frontend import demo, GRADIO_PORT, GRADIO_SERVER_NAME, GRADIO_SHARE, FLASK_API_URL


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("clinical_launcher")


def _run_api() -> None:
    api_port = int(os.environ.get("FLASK_PORT", os.environ.get("APP_PORT", "5000")))
    LOGGER.info("Starting internal Flask API on 127.0.0.1:%d", api_port)
    serve_flask(app, host="127.0.0.1", port=api_port, threads=8)


if __name__ == "__main__":
    api_thread = Thread(target=_run_api, name="clinical-api", daemon=True)
    api_thread.start()
    LOGGER.info("Starting public Gradio UI on %s:%d", GRADIO_SERVER_NAME, GRADIO_PORT)
    LOGGER.info("Frontend API target: %s", FLASK_API_URL)
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        show_api=False,
    )
