import requests

def generate_response(prompt):
    url = "http://127.0.0.1:8000/generate_response"
    params = {"prompt": prompt}
    response = requests.post(url, params=params)
    return response

def upload_file(filepath):
    url = "http://127.0.0.1:8000/upload_files/"
    files = {'files': open(filepath, 'rb')}
    response = requests.post(url, files=files)
    return response

def delete_files(file_ids):
    url = "http://127.0.0.1:8000/delete_files/"
    data = {"file_ids": file_ids}
    response = requests.delete(url, json=data)
    return response

def main():
    print("Welcome to the interactive client for FastAPI operations.")
    print("Use /chat <query> to generate a response.")
    print("Use /upload <filepath> to upload a file.")
    print("Use /delete <file_id1> <file_id2> ... to delete files.")
    print("Type /quit to exit the program.")

    while True:
        user_input = input("> ")
        if user_input.startswith("/quit"):
            print("Exiting program.")
            break
        elif user_input.startswith("/chat"):
            _, prompt = user_input.split(" ", 1)
            response = generate_response(prompt)
            if response.status_code == 200:
                print("Success:", response.json())
            else:
                print("Error:", response.status_code, response.text)
        elif user_input.startswith("/upload"):
            _, filepath = user_input.split(" ", 1)
            try:
                response = upload_file(filepath)
                if response.status_code == 200:
                    print("Success:", response.json())
                else:
                    print("Error:", response.status_code, response.text)
            except Exception as e:
                print("Error:", str(e))
        elif user_input.startswith("/delete"):
            _, *file_ids = user_input.split(" ")
            response = delete_files(file_ids)
            if response.status_code == 200:
                print("Success:", response.json())
            else:
                print("Error:", response.status_code, response.text)
        else:
            print("Unrecognized command. Please try again.")

if __name__ == "__main__":
    main()
