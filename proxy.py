from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # или ["http://localhost:3000"] если хочешь строгость
    allow_methods=["*"],  # или ["POST", "OPTIONS"]
    allow_headers=["*"],  # или ["Content-Type", "Authorization"]
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def mock_chat_completions(request: Request):
    headers = dict(request.headers)
    body = await request.json()
    
    print("\n--- Запрос получен на /v1/chat/completions ---")
    print("📋 Заголовки:")
    for k, v in headers.items():
        print(f"{k}: {v}")
    
    print("\n📦 Тело запроса:")
    print(body)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    file_path = LOGS_DIR / f"request_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)
    
    response = {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": body.get("model", "gpt-4o"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": """```html
<div>
<!-- Тестовый коммент -->
</div>
```"""
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 24,
            "total_tokens": 36
        }
    }    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=40001, reload=True)
