import traceback
import datetime
import asyncio
import base64
import ntpath
import json
import uuid
import sys
import os
import re

from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path

sys.path.append("cv")

from config import DBConnection


app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / 'data/videos'

# Статикой
static_dir = Path("cv/main/static")

# Шаблонам
template_dir = Path("cv/main/templates")

# Подключение статических файлов
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)

# Модель для хранения настроек
class Model:
    def __init__(self):
        self.file_path = None
        self.model_path = PROJECT_ROOT / 'models/yolov10l.pt'
        self.target_video = False
        self.confidence = 0.3
        self.iou = 0.7
        self.directory = UPLOAD_FOLDER
        self.files = os.listdir(self.directory)
        self.processor = None
        self.save_db = True
        self.show_video = False
        self.save_path_line = ''


model = Model()


@app.get("/")
async def show_main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "files": model.files})


@app.post("/", response_class=HTMLResponse)
async def select_video(request: Request, comp_select: str = Form(...)):

    
        
    return templates.TemplateResponse("draw.html", {"request": request})


