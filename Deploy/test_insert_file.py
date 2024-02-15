import requests

url = "http://127.0.0.1:8000/upload_files/"
files = {'files': open('path/to/your/file.pdf', 'rb')}

response = requests.post(url, files=files)

if response.status_code == 200:
    print("Success:")
    print(response.json())
else:
    print("Error:", response.status_code)
    print(response.text)
