from pydantic import BaseModel

class BaseRequest(BaseModel):
    user_sentence:str

class ASRequest(BaseRequest):
    scene:str=''

class BaseResponse(BaseModel):
    code:int=200
    msg:str=''

class DrawActionResponse(BaseResponse):
    ACTION:str='ADD'
    DO:str='None'
    DO_ID:int=-1
    COUNT:int=1
    IO:str='None'
    IO_POS:int=-1
    IO_POS2:int=-1

class IntentResponse(BaseResponse):
    intent:int=0

class SearchResponse(BaseResponse):
    result:str=''
