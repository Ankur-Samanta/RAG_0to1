import requests

url = "http://127.0.0.1:8000/generate_response"
params = {"prompt": "What is the capital of France?"}

response = requests.post(url, params=params)

if response.status_code == 200:
    print("Success:")
    print(response.json())
else:
    print("Error:", response.status_code)
    print(response.text)
