import re,cn2an,torch
import sys
sys.path.append(r"C:\Users\admin\Desktop\AIServer")
class RegularExtract:
    def __init__(self) -> None:
        self.action_pattern=r"(变|换)"
        self.arbic_position_pattern = r"((?P<NUM>\d+))号"
        self.entity_pattern = (
            r"(单键|双键|3键|3元环|4元环|5元环|6元环|7元环|8元环|苯环|三键|本环|3圆环|3元还|4圆环|4元环|5圆环|5元环|6圆环|6元环|7圆环|"
            r"7元环|8圆环|8元环|3人环|4人环|5人环|6人环|7人环|8人环|8元环|环雾丸|环丙烷|环丁烷|黄金丸|缓解完|甲基|甲级|加急|夹击|佳绩|环己烷|继续|回退|提交|清屏|清贫)")

        self.fillter_words= ['上', '的', '吧', '额', '嗯', '啊', '阿', '哦']
    def fillter_slot_value(self,value:str)->str:
        '''
        去除槽位中无意义的词
        '''
        if not isinstance(value, str):
            return value
        for elem in self.fillter_words:
            if elem in value:
                value = value.replace(elem, '')
        return value
    def entity_extract_slot(self,text:str,model,tokenize):
        draw_slot = {
            'code': 200,
            'ACTION': 'ADD',
            'DO': 'None',
            'DO_ID': -1,
            'COUNT': 1,
            'IO': 'None',
            'IO_POS': -1,
            'IO_POS2': -1,
        }
        query=text
        text = cn2an.transform(text)  # 中文转阿拉伯
        text = text.replace(' ', '')  

        #搜索关键字
        action_match = re.search(self.action_pattern, text)  # 动作只有一个
        arbic_position_match = re.findall(
            self.arbic_position_pattern, text)  # 可能有多个位置
        entity_match = re.findall(self.entity_pattern, text)

         # 判断是否为连读，决定有有几个位置字段
        if len(arbic_position_match) == 1:
            # 当提取到的位置只有一个的时候，确认用户不是连读
            if action_match:
                draw_slot=self.inference_position(draw_slot,arbic_position_match)
            else:
                draw_slot['IO_POS'] = int(arbic_position_match[0][0])
        elif len(arbic_position_match) >= 2:
            draw_slot['IO_POS'] = int(arbic_position_match[0][0])
            draw_slot['IO_POS2'] = int(arbic_position_match[1][0])
        elif len(arbic_position_match) == 0:
            draw_slot['IO_POS'] = -1
        
        if len(entity_match) == 0:
            entity_match = self.inference(model, tokenize, text)  # 提取实体
            entity_match = [x[0] for x in entity_match]
            if entity_match is None:
                draw_slot['code']=-1
                draw_slot['DO'] = 'None'
                return draw_slot
    
    def inference(self, model, tokenizer, query: str) -> list[list]:
        schem = {
            0: 'S',
            1: 'M',
            2: 'D',
            3: 'H',
            4: 'R',
            5: 'T',
            6: 'O'
        }
        result = []
        input_sequence, sequence_length = tokenizer.tokenize(query)
        model.eval()
        if torch.cuda.is_available():
            input_sequence = input_sequence.to('cuda')
            sequence_length = sequence_length.to('cuda')
            model = model.cuda()
        output = model(input_sequence, sequence_length)[0]
        output = [schem.get(k) for k in output]
        if 'H' in output and 'T' in output:
            start, end = output.index('H'), output.index('T')
            result.append([query[start:end + 1], start, end])
        if 'S' in output and 'D' in output:
            start, end = output.index('S'), output.index('D')
            result.append([query[start:end + 1], start, end])
        return result
    
    def rebuild(self, slot: dict, last_frame)->dict:
        if slot['DO'] == '继续' and last_frame is not None:
            if last_frame['ACTION'] == 'ADD' and last_frame['IO_POS2'] == - \
                    1:  # 排除上个动作是变换，而被误认为是添加的情况
                last_frame['IO_POS'] = -1
            return last_frame
        elif '返回' in slot['DO']:
            numbers = re.findall(r'\d+', slot['DO'])
            slot['DO'] = '返回'
            slot['COUNT'] = int(numbers[0]) if len(numbers) > 0 else 1
        return slot
    def confirm_action(self, slot):
        if slot['ACTION'] == 'ADD':
            return slot
        elif slot['ACTION'] == 'TRANSFORM':
            if slot['IO_POS2'] == -1:
                slot['ACTION'] = 'ADD'
        return slot