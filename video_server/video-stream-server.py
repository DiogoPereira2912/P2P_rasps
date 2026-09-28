import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable, List

import cv2
import uvicorn
from fastapi import FastAPI, Request, Response, Body
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

RELEASED_CAMS = False
MODEL_RESOLUTION = (1080, 1080)

CAM0 = os.environ.get("CAM0", "/dev/video0")
CAM1 = os.environ.get("CAM1", "/dev/video2")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    global RELEASED_CAMS
    if os.path.exists("rois_url0.yaml") or os.path.exists("rois_url1.yaml"):
        print("ENCONTREI")
        cameras.release()
        RELEASED_CAMS = True
    else:
        pass

    try:
        yield
    except asyncio.exceptions.CancelledError as error:
        print(error.args)
    finally:
        cameras.release()
        print("Camera resource released.")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Camera:
    """
    A class to handle video capture from a camera.
    """

    def __init__(self, url: str | int) -> None:
        """
        Initialize the camera.

        :param camera_index: Index of the camera to use.
        """
        self.cap = cv2.VideoCapture(url)
        self.lock = threading.Lock()
        self.pTime = 0
        self.cTime = 0

    def frame_counter(func: Callable):
        """
        Add a frame counter to the image with detected landmarks.
        """

        def wrapper(self, *args, **kwargs):
            ret, frame = func(self, *args, **kwargs)
            if not ret:
                return ret, frame

            try:
                frame = cv2.resize(frame, MODEL_RESOLUTION)
            except Exception as e:
                print(f"Erro no resize: {e}")

            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            cv2.putText(
                frame,
                str(int(fps)),
                (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                3,
            )
            return ret, frame

        return wrapper

    @frame_counter
    def get_frame(self) -> bytes:
        """
        Capture a frame from the camera.

        :return: JPEG encoded image bytes.
        """
        with self.lock:
            return self.cap.read()

    def get_frame_jpeg(self) -> bytes:
        """
        Convert a captured frame from the camera to JPEG image.

        :return: JPEG encoded image bytes.
        """
        ret, frame = self.get_frame()
        if not ret:
            return b""

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            return b""

        return jpeg.tobytes()

    def release(self) -> None:
        """
        Release the camera resource.
        """
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()


class CameraManager:
    """
    A class for handling multiple cameras for image capture.
    """

    def __init__(self, urls: str | int | List[str | int]) -> None:
        self.camera_urls = urls
        self.cameras = [Camera(url) for url in urls]
        self.released_cameras_status = [False] * len(self.cameras)

    def get_camera_by_id(self, id: int) -> Camera:
        return self.cameras[id]

    def release(self) -> None:
        for camera in self.cameras:
            camera.release()

    def release_camera_by_id(self, id: int) -> None:
        self.cameras[id].release()
        self.released_cameras_status[id] = True
        print(self.released_cameras_status)

    def all_cams_released(self) -> bool:
        return all(self.released_cameras_status)

async def gen_frames(cam_id: int) -> AsyncGenerator[bytes, None]:
    """
    An asynchronous generator function that yields camera frames.

    :yield: JPEG encoded image bytes.
    """
    try:
        while True:
            camera = cameras.get_camera_by_id(cam_id)
            frame = camera.get_frame_jpeg()
            if frame:
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            else:
                break
            await asyncio.sleep(0)
    except (asyncio.CancelledError, GeneratorExit):
        print("Frame generation cancelled.")
    finally:
        print("Frame generator exited.")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.get("/video/{cam_id}")
async def video_feed(cam_id: int) -> StreamingResponse:
    """
    Video streaming route.

    :return: StreamingResponse with multipart JPEG frames.
    """
    return StreamingResponse(
        gen_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/snapshot/{cam_id}")
async def snapshot(cam_id: int) -> Response:
    """
    Snapshot route to get a single frame.

    :return: Response with JPEG image.
    """
    camera = cameras.get_camera_by_id(cam_id)
    frame = camera.get_frame()
    if frame:
        return Response(content=frame, media_type="image/jpeg")
    else:
        return Response(status_code=404, content="Camera frame not available.")

# --- MEU --- #

@app.post("/upload_roi")
async def upload_roi_config(video_key: str,  yaml_content: str = Body(..., media_type="text/plain")):
    """
    Recebe o conteúdo de um ficheiro YAML e guarda-o localmente no Pi.
    Dá release nas cameras uma de cada vez 
    Torna a flag RELEASED_CAMS = TRUE, para retornar status:ok no healthcheck
    """

    filename = f"rois_{video_key}.yaml"

    cam_id = int(video_key[-1])  
    print("Cam ID:", cam_id)
    
    save_path = os.path.join(os.getcwd(), filename)

    with open(save_path, "w") as f:
        f.write(yaml_content)

    print(f"Configuração de ROI recebida e guardada em: {save_path}")

    cameras.release_camera_by_id(cam_id)    

    if cameras.all_cams_released():
        print("All cameras released")
        global RELEASED_CAMS
        RELEASED_CAMS = True
    
    return JSONResponse(
        status_code=200,
        content={"status": "success"}
    )



@app.get("/health")
def health():
    if RELEASED_CAMS:
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "cameras": "released"}
        )
    else:
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content={"status": "not_ready", "cameras": "in_use"}
        )

async def main():
    """
    Main entry point to run the Uvicorn server.
    """
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)

    # Run the server
    await server.serve()

if __name__ == "__main__":
    cameras = CameraManager([CAM0, CAM1])
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")