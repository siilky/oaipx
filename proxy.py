import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Callable, Any, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, RedirectResponse

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y%m%d %H:%M.%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# get environment variables

log_json_headers = os.environ.get("LOG_JSON_HEADERS", "false").lower() in ['true', '1', 'yes']

logging.info(f'Logging JSON headers: {log_json_headers}')

#
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['POST', 'OPTIONS', 'GET'],
    allow_headers=['*'],
)

AISTUDIO_URL = 'https://generativelanguage.googleapis.com/v1beta/models'
OPENROUTER_URL = 'https://openrouter.ai/api/v1'
OPENROUTER_COMPLETIONS_URL = OPENROUTER_URL + '/chat/completions'
OPENROUTER_MODELS_URL = OPENROUTER_URL + '/models'
DEFAULT_MODEL = 'openrouter/auto'


def print_json(data, depth=1, current_level=1):
    """ Prints JSON data in a structured format, limiting the depth of printed objects with current_level. """

    if not log_json_headers:
        return

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
    logging.info(json.dumps(result, indent=4, default=str))


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
    # Reasoning tokens will appear in the reasoning field of each message.
    # so there's no much reason to turn it off at the api request
    pass


def empty_handler(request: dict, value):
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
    'autoroute': empty_handler,
    'or_key': empty_handler,
    'google_key': empty_handler,
}


def modify_break_sys(messages: list, message_index: int, entry_pos: int):
    # cutoff the system message at the position of the <<break_sys>> tag and insert user message after (all possible) system's
    text = messages[message_index].get('content', '')
    messages[message_index]['content'] = text[:entry_pos].rstrip()  # Remove trailing whitespace after the break

    while messages[message_index].get('role') == 'system':
        message_index += 1

    # Insert a new user message after the system messages
    messages.insert(message_index, {
        'role': 'user',
        'content': text[entry_pos:].lstrip()  # Remove leading whitespace after the break
    })


MODIFIERS = {
    'break_sys': modify_break_sys,
}


###

def extract_and_remove_commands(text: str, handlers: Dict[str, Callable[[Dict], Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts parameters based on handler keys and removes them from the text.

    This function finds tags like <name>, <name=>, or <name=value>. If 'name'
    is a key in `handlers`, the tag is treated as a command and
    removed from the text.

    Returns:
        A tuple containing the cleaned text and extracted parameters with their raw values.
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
                message['content'] = text  # Update the message content with the cleaned text

                for command, value in commands.items():
                    if command in valid_commands:
                        try:
                            valid_commands[command](request, value)
                        except Exception as e:
                            msg = f"Failed to process command '{command}': {e}"
                            logging.error(msg)
                            # raise HTTPException(status_code=400, detail=msg)
            return commands
    return {}


def process_modifiers(valid_modifiers: Dict[str, Callable[[Dict], Any]], request: Dict) -> Dict[str, Any]:
    pattern = re.compile(r'<<([a-zA-Z0-9_]+)>>', re.IGNORECASE)

    messages = request.get('messages')

    message_index = 0
    while message_index < len(messages):
        message = messages[message_index]
        if message.get('role') == 'system' and 'content' in message:

            text = message['content']
            match = pattern.search(text)
            if match:
                message['content'] = text[:match.start()] + text[match.end():]  # Remove the modifier tag
                modifier = match.group(1).lower()
                if modifier in valid_modifiers:
                    try:
                        valid_modifiers[modifier](messages, message_index, match.start())
                    except Exception as e:
                        msg = f"Failed to process modifier '{modifier}': {e}"
                        logging.error(msg)
                        # raise HTTPException(status_code=400, detail=msg)

        message_index += 1


def asResponse(text: str, finish_reason: str = None):
    choice = {
        'message': {
            'role': 'assistant',
            'content': text,
        },
    }

    if finish_reason:
        choice['finish_reason'] = finish_reason

    return {
        'choices': [choice]
    }


def merge_thinking(text: str, reasoning: str) -> str:
    if reasoning:
        return f'<think>{reasoning}\n---\n</think>\n{text}'
    return text


async def _proxy_aistudio_request(headers: dict, body: dict, commands: dict):
    model = body.get('model', DEFAULT_MODEL)
    if model.startswith('google/'):
        model = model.replace('google/', '', 1)

    api_key = headers.get('Authorization')
    api_key = api_key.replace('Bearer ', '', 1)
    headers.pop('Authorization', None)

    # convert messages
    contents = []
    system_instructions = []

    for message in body.get('messages', []):
        if 'content' in message:
            role = message.get('role', 'user').lower()

            if role == 'system':
                system_instructions.append({'text': message['content']})
                continue

            if role != 'user':
                role = 'model'

            contents.append({
                'role': role,
                'parts': [{'text': message['content']}]
            })

    config = {}
    if 'temperature' in body:
        config['temperature'] = body['temperature']
    if 'top_p' in body:
        config['topP'] = body['top_p']
    if 'top_k' in body:
        config['topK'] = body['top_k']
    if 'presence_penalty' in body:
        config['presencePenalty'] = body['presence_penalty']
    if 'max_completion_tokens' in body:
        config['maxOutputTokens'] = body['max_completion_tokens']
    if 'stop' in body:
        config['stopSequences'] = body['stop']
    if 'reasoning' in body and isinstance(body['reasoning'], dict):
        thinking_config = {}
        if not body['reasoning'].get('enabled', False):
            # disable thinking if disabled, otherwise unset it to use default
            thinking_config['thinkingBudget'] = 0
        if commands.get('show_thinking', False):
            thinking_config['includeThoughts'] = True
        config['thinkingConfig'] = thinking_config

    url = f'{AISTUDIO_URL}/{model}:generateContent?key={api_key}'
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            url,
            headers=headers,
            json={
                'contents': contents,
                'systemInstruction': {'parts': system_instructions},
                'safetySettings': [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "OFF"}
                ],
                'generationConfig': config,
            }
        )

        response.raise_for_status()
        reply = response.json()

        print_json(reply, 3)

        meta = reply.get('usageMetadata', {})
        candidates_str = f"text {meta['candidatesTokenCount']}tk" if 'candidatesTokenCount' in meta else ''
        reasoning_str = f", reasoning {meta['thoughtsTokenCount']}tk" if 'thoughtsTokenCount' in meta else '❌'

        logging.info(f"GAI: prompt {meta['promptTokenCount']}tk => {candidates_str}{reasoning_str}")

        completion = ''
        reasoning = ''
        finish_reason = ''

        if 'candidates' in reply and len(reply['candidates']) > 0:
            candidate = reply['candidates'][0]

            if 'content' in candidate and 'parts' in candidate['content']:
                for part in candidate['content']['parts']:
                    if 'thought' in part and part['thought']:
                        reasoning += part['text'] + '\n'
                    else:
                        completion += part['text'] + '\n'

            # merge reasoning into the content if show_thinking is True
            show_thinking = commands.get('show_thinking', False)
            if show_thinking and reasoning:
                completion = merge_thinking(completion, reasoning)

            finish_reason = candidate.get('finishReason', '')
        else:
            # If no candidates are returned, form an error
            if feedback := reply.get('promptFeedback'):
                completion = json.dumps(feedback, indent=2)
                finish_reason = feedback['blockReason']
            else:
                completion = "❌ Invalid response from API: 'candidates' field missing or empty"

        return asResponse(completion, finish_reason)


async def _proxy_openrouter_request(headers: dict, body: dict, commands: dict):
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            OPENROUTER_COMPLETIONS_URL,
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        reply_data = response.json()

        if 'choices' in reply_data and len(reply_data['choices']) > 0:
            if 'message' not in reply_data['choices'][0]:
                logging.error("❌ Invalid response from proxy: 'message' field missing in choices")
                raise HTTPException(status_code=502, detail="Invalid response from proxy: 'message' field missing")

            message = reply_data['choices'][0]['message']

            if 'content' not in message:
                logging.error("❌ Invalid response from proxy: 'content' field missing in message")
                raise HTTPException(status_code=502, detail="Invalid response from proxy: 'content' field missing in message")

            completion = message['content']
            reasoning = message.get('reasoning')

            usage = reply_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            logging.info(
                f"[OpR: prompt {prompt_tokens}tk => text {completion_tokens}tk ({len(completion)}){(f', reasoning: {len(reasoning)}' if reasoning else '')}")
            print_json(reply_data, 2)

            # merge reasoning into the content if show_thinking is True
            show_thinking = commands.get('show_thinking', False)
            if show_thinking and reasoning:
                message['content'] = merge_thinking(message['content'], reasoning)

        return reply_data


async def _proxy_request(headers: dict, body: dict):
    logging.info(f"--- input {len(body.get('messages', []))} messages ---")

    if 'model' not in body:
        body['model'] = DEFAULT_MODEL

    commands = process_commands(COMMANDS, body)
    process_modifiers(MODIFIERS, body)

    print_json(body)

    # strip the <think> tags from the content (only from model messages)
    if commands.get('show_thinking'):
        for message in body['messages']:
            if message.get('role') in ['model', 'assistant'] and 'content' in message:
                message['content'] = re.sub(r'<think>.*?</think>', '', message['content'], flags=re.DOTALL)

    try:
        if commands.get('autoroute') and (body['model'].startswith('google/') or body['model'].startswith('gemini-')):
            # when autoroute is enabled and we are on Google Gemini or Google AI Studio, use google_key option
            if google_key := commands.get('google_key'):
                headers['Authorization'] = f'Bearer {google_key}'

            return await _proxy_aistudio_request(headers, body, commands)
        else:
            # when no autoroute we take OR key first
            if key := (commands.get('or_key') or commands.get('google_key')):
                headers['Authorization'] = f'Bearer {key}'

            return await _proxy_openrouter_request(headers, body, commands)

    except httpx.HTTPStatusError as e:
        return asResponse(f"❌ Proxy Error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        return asResponse(f"❌ Proxy Error: {str(e)}")


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
        logging.error(f"❌ Models request failed: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch models from OpenRouter")
    except Exception as e:
        logging.error(f"❌ Models processing error: {e}")
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
    reply = await _proxy_request(headers, body)
    return reply


@app.get("/chat/completions")
async def completions_redirect():
    return RedirectResponse(url="/", status_code=302)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)
