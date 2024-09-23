#元数据
class MetaData:
    def __init__(self,intent:int,slot:dict) -> None:
        self.intent=intent
        self.slot=slot
class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        """检查栈是否为空"""
        return len(self.items) == 0

    def push(self, item:MetaData):
        """向栈中添加一个元素"""
        self.items.append(item)

    def pop(self)->MetaData:
        """从栈中移除并返回顶部元素"""
        if not self.is_empty():
            return self.items.pop()
        return None  # 如果栈为空，返回 None

    def peek(self)->MetaData:
        """返回栈顶元素，但不从栈中移除它"""
        if not self.is_empty():
            return self.items[-1]
        return None  # 如果栈为空，返回 None

    def size(self):
        """返回栈的大小"""
        return len(self.items)

class DialogueManager:
    def __init__(self) -> None:
        self.frame_buffer=Stack()

    def add_frame(self,frame:MetaData)->None:
        self.frame_buffer.append(frame)

    def get_frame(self)->MetaData:
        return self.frame_buffer.peek()

    def remove_last_frame(self)->None:
        self.frame_buffer.pop()
