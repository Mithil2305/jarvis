
import requests
import json

url = "http://127.0.0.1:8080/v1/chat/completions"

payload = {
    "model": "gemma-4",
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly: Hello Jarvis!"
        }
    ],
    "max_tokens": 128,
    "temperature": 0.2,
    "stream": False
}

try:
    response = requests.post(url, json=payload, timeout=180)

    print("HTTP STATUS:", response.status_code)
    response.raise_for_status()

    data = response.json()
    print("\nRAW RESPONSE:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        print("\nMESSAGE CONTENT:", repr(message.get("content")))
        print("FINISH REASON:", choices[0].get("finish_reason"))
    else:
        print("\nNo choices returned.")

except Exception as error:
    print("ERROR:", repr(error))
    if getattr(error, "response", None) is not None:
        print(error.response.text)
