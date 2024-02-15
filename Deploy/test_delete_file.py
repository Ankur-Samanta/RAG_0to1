import requests

url = "http://127.0.0.1:8000/delete_files/"
data = {"file_ids": ["your-file-id-1", "your-file-id-2"]}  # Replace with actual file IDs to delete

response = requests.delete(url, json=data)

if response.status_code == 200:
    print("Success:")
    print(response.json())
else:
    print("Error:", response.status_code)
    print(response.text)
