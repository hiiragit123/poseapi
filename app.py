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
    Point = "Point"

@app.post('/api/{predict_object}')
async def judge_image(predict_object:PredictName, files: List[UploadFile] = File(...)):
    ans = await files[0].read()
    input = await files[1].read()
    ans = np.fromstring(ans,np.uint8)
    input = np.fromstring(input,np.uint8)
    ai,ii = pose_app.get_image(ans,input)
    ap,ip = pose_app.get_pose(ai,ii)
    dist = pose_app.get_dist(ap,ip)
    similarity = pose_app.get_similarity(dist)
    point = pose_app.judge(similarity)
    res = {
            "Point" : point
        }
    return res