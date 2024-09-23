"""
descripition:定义接口
update:2024.9.20
"""

from fastapi import APIRouter, Depends
import sys, os

sys.path.append(r"C:\Users\admin\Desktop\AIServer")
from modules.Intent_routing_center import IntentRoutingCenter
from modules.draw_instruct import DrawInstruct
from settings.schem import IntentResponse, DrawActionResponse,SearchResponse
from modules.load_modules import load_intent_model, load_draw_instruct_model

router = APIRouter()

prefix = "AiServer/"


@router.get(prefix + "get_intent")
def get_user_intent(
    sentence: str, intent_predicter: IntentRoutingCenter = Depends(load_intent_model)
) -> IntentResponse:
    intent = intent_predicter.predict(sentence)
    return IntentResponse(intent=intent)


@router.get(prefix + "drawScene")
def get_action_instruction(
    query: str, draw_instructor: DrawInstruct = Depends(load_draw_instruct_model)
) -> DrawActionResponse:
    slot = draw_instructor.instruction_analysis(query)

    if slot["code"] != 200:
        return DrawActionResponse(code=slot["code"], msg=slot["msg"])
    else:
        return DrawActionResponse(
            ACTION=slot["ACTION"],
            DO=slot["DO"],
            DO_ID=slot["DO_ID"],
            COUNT=slot["COUNT"],
            IO=slot["IO"],
            IO_POS=slot["IO_POS"],
            IO_POS2=slot["IO_POS2"],
        )

@router.get(prefix + "search")
def search_engine(
    query:str
)->SearchResponse:
    pass