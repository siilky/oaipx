import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Any, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

def set_model(request: dict, model):
    if isinstance(model, str) and model:
        request['model'] = model


def set_temperature(request: dict, temperature):
    if isinstance(temperature, (int, float)):
        request['temperature'] = temperature
    else:
        request.pop('temperature', None)


def set_top_p(request: dict, top_p):
    if isinstance(top_p, (int, float)):
        request['top_p'] = top_p
    else:
        request.pop('top_p', None)


def set_top_k(request: dict, top_k):
    if isinstance(top_k, (int, float)):
        request['top_k'] = top_k
    else:
        request.pop('top_k', None)


def set_presence_penalty(request: dict, presence_penalty):
    if isinstance(presence_penalty, (int, float)):
        request['presence_penalty'] = presence_penalty
    else:
        request.pop('presence_penalty', None)


def set_max_tokens(request: dict, max_tokens):
    if isinstance(max_tokens, int) and max_tokens > 0:
        request['max_completion_tokens'] = max_tokens
    else:
        request.pop('max_completion_tokens', None)


def set_stop(request: dict, stop):
    if isinstance(stop, str) and stop:
        request['stop'] = [stop]
    elif isinstance(stop, list):
        request['stop'] = [s for s in stop if isinstance(s, str) and s]
    else:
        request.pop('stop', None)


def set_thinking(request: dict, thinking):
    if isinstance(thinking, bool):
        request['reasoning'] = {
            'enabled': thinking,
        }
    elif isinstance(thinking, str) and thinking in ['high', 'medium', 'low']:
        request['reasoning'] = {
            'enabled': True,
            'effort': thinking,
        }
    else:
        request.pop('thinking', None)


def set_show_thinking(request: dict, show_thinking):
    # Reasoning tokens will appear in the reasoning field of each message.#
    # so there's no much reason to turn it off at the api request
    pass


# commands are lowercase, but we allow uppercase in the text
COMMANDS = {
    'model': set_model,
    'temperature': set_temperature,
    'top_p': set_top_p,
    'top_k': set_top_k,
    'presence_penalty': set_presence_penalty,
    'max_tokens': set_max_tokens,
    'stop': set_stop,
    'thinking': set_thinking,
    'show_thinking': set_show_thinking,
}


###

def extract_and_remove_commands(text: str, handlers: Dict[str, Callable[[Dict], Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts parameters based on handler keys and removes them from the text.

    This function finds tags like <name>, <name=>, or <name=value>. If 'name'
    is a key in `handlers`, the tag is treated as a command and
    removed from the text.

    Returns:
        A tuple containing the cleaned text and extracted
        parameters with their raw values.
    """

    pattern = re.compile(r'<([a-zA-Z0-9_]+)((?:=[^>]*)?)>', re.IGNORECASE)

    valid_commands = set(handlers.keys())

    params = {}

    def replacer(match):
        full_tag = match.group(0)
        name = match.group(1).lower()
        value_part = match.group(2)

        if name not in valid_commands:
            return full_tag  # Not a command we handle, leave it.

        if value_part is None or value_part == "":
            # Format: <name> -> True
            value = True
        elif value_part == "=":
            # Format: <name=> -> ""
            value = ""
        else:
            # Format: <name=value> -> "value" (as a string)
            value = value_part[1:]  # Remove leading '='
            if value.casefold() in ['true', 'yes', 'on']:
                value = True
            elif value.casefold() in ['false', 'no', 'off']:
                value = False
            elif '.' in value:
                if value.replace('.', '', 1).isdigit():
                    # Convert to float if it looks like a number
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            elif value.isdigit():
                # Convert to int if it looks like a number
                value = int(value)
            elif value.startswith('[') and value.endswith(']'):
                # Convert to list if it looks like a list
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass

        params[name] = value
        return ""  # Remove the tag from the text

    processed_text = pattern.sub(replacer, text)
    return processed_text, params


def process_commands(valid_commands: Dict[str, Callable[[Dict], Any]], request: Dict) -> Dict[str, Any]:
    for message in request['messages']:
        if message.get('role') == 'system' and 'content' in message:
            text, commands = extract_and_remove_commands(message['content'], valid_commands)

            if commands:
                # Update the message content with the cleaned text
                message['content'] = text

                for command, value in commands.items():
                    if command in COMMANDS:
                        try:
                            COMMANDS[command](request, value)
                        except Exception as e:
                            print(f"Error processing command '{command}': {e}")
                            # raise HTTPException(status_code=400, detail=f"Invalid command: {command_name}")
            return commands
    return {}


async def _proxy_to_openrouter(headers: dict, body: dict):
    if 'model' not in body:
        body['model'] = DEFAULT_MODEL
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}] --- {body['model']}")

    commands = process_commands(COMMANDS, body)

    show_thinking = commands.get('show_thinking', False)

    print_json(body)

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            openrouter_response = await client.post(
                OPENROUTER_COMPLETIONS_URL,
                headers=headers,
                json=body,
            )
            openrouter_response.raise_for_status()
            reply_data = openrouter_response.json()

            if 'choices' in reply_data and len(reply_data['choices']) > 0:
                if 'message' not in reply_data['choices'][0]:
                    print("❌ Invalid response from proxy: 'message' field missing in choices")
                    raise HTTPException(status_code=502, detail="Invalid response from proxy: 'message' field missing")

                message = reply_data['choices'][0]['message']

                if 'content' not in message:
                    print("❌ Invalid response from proxy: 'content' field missing in message")
                    raise HTTPException(status_code=502, detail="Invalid response from proxy: 'content' field missing in message")

                completion = message['content']
                reasoning = message.get('reasoning')

                usage = reply_data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)

                print(
                    f"[{datetime.now().strftime('%Y%m%d_%H%M%S.%f')}] OpR prompt {prompt_tokens}tk: text {completion_tokens}tk ({len(completion)}){(f' reasoning: {len(reasoning)}' if reasoning else '')}")
                print_json(reply_data, 2)

                # merge reasoning into the content if show_thinking is True
                if show_thinking and reasoning:
                    message['content'] = f'<think>{reasoning}</think>\n\n---\n{completion}'

            return reply_data

    except httpx.HTTPStatusError as e:
        print(f"❌ Proxy Error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        print(f"❌ Proxy Error: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch from Proxy")


def copy_headers(request: Request) -> dict:
    """
    Copies relevant headers from the request to the response.
    """

    headers = {}
    for header in ['Origin', 'Referer', 'User-Agent', 'HTTP-Referer', 'X-Title', 'Content-Type']:
        if header in request.headers:
            headers[header] = request.headers[header]
    return headers


###

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / 'index.html')


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
    headers = copy_headers(request)
    # OpenAI uses 'Authorization' header for authentication
    if "Authorization" in request.headers:
        headers['Authorization'] = request.headers["Authorization"]
    else:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    body = await request.json()
    reply = await _proxy_to_openrouter(headers, body)
    return reply


# anthropic API check
# The /v1/complete endpoint belongs to Anthropic's legacy Completions API.
# The current recommended API is the Messages API, which uses the /v1/messages endpoint.
@app.post('/complete')
async def complete_default(request: Request):
    return {
        'completion': ' Hello! How can I help you today?',
        'stop_reason': 'stop_sequence',
    }


# anthropic API endpoint
@app.post('/messages')
async def messages(request: Request):
    headers = copy_headers(request)
    # anthropic api uses 'x-api-key' header for authentication
    if "X-Api-key" in request.headers or "x-api-key" in request.headers:
        headers['Authorization'] = 'Bearer ' + (request.headers.get('x-api-key') or request.headers.get('X-Api-key'))
    else:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    # Map Anthropic's 'stop_sequences' to OpenAI's 'stop'
    body = await request.json()
    if 'stop_sequences' in body:
        body['stop'] = body['stop_sequences']

    reply = await _proxy_to_openrouter(headers, body)
    print_json(reply, 2)

    # Transform OpenRouter response to Anthropic format
    if 'choices' in reply and len(reply['choices']) > 0:
        first_choice = reply['choices'][0]
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
            'id': reply['id'],
            'type': 'message',
        }
    else:
        raise HTTPException(status_code=502, detail="Invalid response from proxy")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)
