import json
from channels.generic.websocket import AsyncWebsocketConsumer

class TrainingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # 그룹에 클라이언트 추가
        self.training_id = self.scope['url_route']['kwargs']['training_id']
        self.group_name = f'training_{self.training_id}'
        
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        await self.accept()
    
    async def disconnect(self, close_code):
        # 그룹에서 클라이언트 제거
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        # 클라이언트로부터 받은 메시지 처리 (필요시)
        pass
    
    async def training_update(self, event):
        # 모델 훈련 업데이트 메시지를 클라이언트에 전송
        message = event['message']
        
        await self.send(text_data=json.dumps({
            'type': 'training_update',
            'message': message
        }))