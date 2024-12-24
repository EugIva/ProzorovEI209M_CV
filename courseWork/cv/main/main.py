import uvicorn
import asyncio
import ntpath
import cv2
import sys
import os

from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.append("/app/cv")

from utils import VideoHandler


app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / 'data/videos/raw'

# Статика
static_dir = Path("cv/main/static")

# Шаблоны
template_dir = Path("cv/main/templates")

# Подключение статических файлов
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)

# Модель для хранения настроек
class Model:
    def __init__(self):
        self.file_path = None
        self.model_path = PROJECT_ROOT / 'cv/models/yolov10l.pt'
        self.target_video = True
        self.confidence = 0.3
        self.iou = 0.7
        self.directory = UPLOAD_FOLDER
        self.files = os.listdir(self.directory)
        self.processor = None
        self.save_db = True
        self.show_video = True
        self.image_path = 'data/images/'


model = Model()


def save_frame(video_path, frame_num, result_path):
    '''
    Функция для сохранения кадра видео
    '''
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(result_path, frame)


async def process_video_async(
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str,
        confidence_threshold: float,
        iou_threshold: float,
        show_video: bool,
        save_db: bool
        ):
    """
    Вызов класса для работы с полигонами
    """

    model.processor = VideoHandler(
        source_weights_path=source_weights_path,
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        show_video=show_video,
        save_db=save_db
    )
    await asyncio.to_thread(model.processor.process_video)


@app.get("/")
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "files": model.files})


@app.post("/upload_video", response_class=HTMLResponse)
def upload_video(request: Request, files: list[UploadFile] = File(...)):
    
    for uploaded_file in files:
        file_name = uploaded_file.filename
        file_path = UPLOAD_FOLDER / file_name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.file.read())

        model.file_path = file_path
        path_image = PROJECT_ROOT / f"{model.image_path}{ntpath.basename(file_path).split('.')[0]}.jpg"
        
        save_frame(file_path, 1, path_image)

        model.files = os.listdir(model.directory)
        
    return templates.TemplateResponse("index.html", {"request": request, "files": model.files})


@app.post("/run_analysis", response_class=HTMLResponse)
async def run_video_detection_processing(request: Request, background_tasks: BackgroundTasks):

    data = await request.json()     # Получаем тело запроса
    # data = json.loads(body)         # Переводим в dict
    model.file_path = UPLOAD_FOLDER / data['file_name']

    target_video_path = PROJECT_ROOT / f"/data/featured/{ntpath.basename(model.file_path).split('.')[0]}.mp4"
    
    background_tasks.add_task(
        process_video_async,
        model.model_path,       # source_weights_path
        model.file_path,        # source_video_path
        target_video_path,      # target_video_path
        model.confidence,       # confidence_threshold
        model.iou,              # iou_threshold
        model.show_video,       # show_video
        model.save_db,          # save_db
        )
    
    return templates.TemplateResponse("video_feed.html", {"request": request})


@app.get("/video_feed", response_class=HTMLResponse)
async def video_feed(request: Request):
    """
    Страница отображения потока видео
    """

    return templates.TemplateResponse("video_feed.html",
                                      {"request": request})


@app.get("/start_stream")
async def start_sstream(request: Request):
    """
     Передача фреймов для показа трансляции обработки видео
     """
    return StreamingResponse(model.processor.get_frame(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stop_stream", response_class=JSONResponse)
async def stop_stream(request: Request):
    """
    Остановка процесса обработки видео и трансляции
    """
    model.processor.stop_stream()
    return JSONResponse(content={"status": "Stream stopped."}, status_code=200)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)