from fastapi import FastAPI
from fastapi import UploadFile, File
from typing import List
import main
import numpy as np
from fastapi.responses import FileResponse
from enum import Enum

app = FastAPI()
pose_app = main.pose_app()
class PredictName(str,Enum):
    Notation1 = "Notation1"

@app.post('/api/{predict_object}')
async def judge_image(predict_object:PredictName, file: UploadFile = File(...)):
    input = await file.read()
    #ans = np.fromstring(ans,np.uint8)
    input = np.fromstring(input,np.uint8)
    if predict_object.value == "Notation1":
        ans = "ans/test1.png"
    ai,ii = pose_app.get_image(ans,input)
    ap,ip = pose_app.get_pose(ai,ii)
    dist = pose_app.get_dist(ap,ip)
    similarity = pose_app.get_similarity(dist)
    point = pose_app.judge(similarity)
    res = {
            "Point" : point
        }
    return res