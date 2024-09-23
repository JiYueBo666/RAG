'''
descripition:入口文件
update: 2024.9.20
'''
import uvicorn
from fastapi import FastAPI
from api.api import router
from utils.utils import get_logger

logger=get_logger()

if __name__ == '__main__':
    
    app=FastAPI()
    app.include_router(router)
    uvicorn.run(app,host="0.0.0.0", port=8000)
