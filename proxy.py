import sys

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import httpx
from starlette.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['POST', 'OPTIONS', 'GET'],
    allow_headers=['*'],
)

OPENROUTER_URL = 'https://openrouter.ai/api/v1'
OPENROUTER_COMPLETIONS_URL = OPENROUTER_URL + '/chat/completions'
OPENROUTER_MODELS_URL = OPENROUTER_URL + '/models'
DEFAULT_MODEL = 'openrouter/auto'


def print_json(data, depth=1, current_level=1):
    def process(obj, level):
        if isinstance(obj, dict):
            if level > depth:
                return f"<dict: {len(obj)} keys>"
            return {k: process(v, level + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            if level > depth:
                return f"<list: {len(obj)} items>"
            return [process(item, level + 1) for item in obj]
        else:
            return obj

    result = process(data, current_level)
    print(json.dumps(result, indent=4, default=str))


###

async def _proxy_to_openrouter(model: str, auth: str, body: dict, headers: dict):
    # Construct payload for OpenRouter, which is OpenAI-compatible.
    payload = {
        'model': model,
        'messages': body.get('messages', []),
        'temperature': body.get("temperature"),
        'top_p': body.get('top_p'),
        "max_tokens": body.get("max_tokens"),
        'stream': body.get('stream', False),
    }
    # the only parameters that are valid (presence_penalty is hardcoded to 0.8, 'Repetition Penalty' just doesn't exists for openai)
    if 'frequency_penalty' in body:
        payload['frequency_penalty'] = body['frequency_penalty']
    if 'top_p' in body:
        payload['top_p'] = body['top_p']

    # Map Anthropic's 'stop_sequences' to OpenAI's 'stop'
    if 'stop_sequences' in body:
        payload['stop'] = body['stop_sequences']
    else:
        payload['stop'] = body.get('stop')

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            openrouter_response = await client.post(
                OPENROUTER_COMPLETIONS_URL,
                headers={
                    "Authorization": auth,
                    "Content-Type": "application/json",
                    "http-referrer": "https://chub.ai",  # Optional: referrer spoofing
                    "x-title": "Chub AI Proxy"  # Optional: client identification
                },
                json=payload
            )
            openrouter_response.raise_for_status()
            openrouter_data = openrouter_response.json()

            if 'choices' in openrouter_data and len(openrouter_data['choices']) > 0:
                first_choice = openrouter_data['choices'][0]
                message = first_choice.get('message', {})
                completion = message.get('content', '')
                reasoning = message.get('reasoning', '')

                usage = openrouter_data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)

                print(
                    f"[{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}] OpR completion prompt {prompt_tokens}: text {completion_tokens} tk ({len(completion)}{(f' reasoning: {len(reasoning)}' if reasoning else '')})")

            return openrouter_data

    except httpx.HTTPStatusError as e:
        print(f"❌ Proxy Error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        print(f"❌ Proxy Error: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch from Proxy")


###

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / 'index.html')


###

@app.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(OPENROUTER_MODELS_URL)
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        print(f"❌ Models request failed: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch models from OpenRouter")
    except Exception as e:
        print(f"❌ Models processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing models")


@app.post("/chat/completions")
async def completions(request: Request):
    headers = dict(request.headers)
    body = await request.json()

    model = body.get('model', '')
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}] --- {model}")
    # for k, v in headers.items():
    #    print(f"{k}: {v}")
    print_json(body)

    # OpenAI uses 'Authorization' header for authentication
    if "Authorization" in headers or "authorization" in headers:
        auth_header = headers.get("authorization") or headers.get("Authorization")
    else:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    openrouter_data = await _proxy_to_openrouter(model, auth_header, body, headers)
    print_json(openrouter_data, 2)
    return openrouter_data


# anthropic API check
# The /v1/complete endpoint belongs to Anthropic's legacy Completions API.
# The current recommended API is the Messages API, which uses the /v1/messages endpoint.
@app.post('/complete')
async def complete_default(request: Request):
    return await complete(request)


@app.post('/{model:path}/complete')
async def complete(request: Request, model: str):
    return {
        'completion': ' Hello! How can I help you today?',
        'stop_reason': 'stop_sequence',
    }


@app.post('/messages')
async def messages_default(request: Request):
    return await messages(request, DEFAULT_MODEL)


# @app.post("/chat/completions")
@app.post('/{model:path}/messages')  # anthropic API endpoint
async def messages(request: Request, model: str):
    headers = dict(request.headers)
    body = await request.json()

    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}] --- {model}")
    # for k, v in headers.items():
    #    print(f"{k}: {v}")
    print_json(body)

    auth_header = ''
    # anthropic api uses 'x-api-key' header for authentication
    if "X-Api-key" in headers or "x-api-key" in headers:
        auth_header = 'Bearer ' + (headers.get('x-api-key') or headers.get('X-Api-key'))
    else:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    openrouter_data = await _proxy_to_openrouter(model, auth_header, body, headers)
    print_json(openrouter_data, 2)

    # Transform OpenRouter response to Anthropic format
    if 'choices' in openrouter_data and len(openrouter_data['choices']) > 0:
        first_choice = openrouter_data['choices'][0]
        stop_reason = first_choice.get('finish_reason', 'unknown')
        message = first_choice.get('message', {})
        completion = message.get('content', '')

        return {
            'content': [
                {
                    'type': 'text',
                    'text': completion
                }
            ],
            'stop_reason': stop_reason,
            'id': openrouter_data['id'],
            'type': 'message',
        }
    else:
        raise HTTPException(status_code=502, detail="Invalid response from proxy")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)
