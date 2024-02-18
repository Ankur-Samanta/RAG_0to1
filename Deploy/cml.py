import requests

def generate_response(prompt):
    url = "http://127.0.0.1:8000/generate_response"
    params = {"prompt": prompt}
    response = requests.post(url, params=params)
    return response

def upload_files(filepaths_with_nicknames):
    url = "http://127.0.0.1:8000/upload_files/"
    files = []
    # Prepare the files and nicknames for multipart/form-data
    for file_nickname_pair in filepaths_with_nicknames:
        filepath, nickname = file_nickname_pair.split(':')
        try:
            file_handle = open(filepath, 'rb')
            files.append(('files', (nickname, file_handle)))
        except Exception as e:
            print(f"Error opening {filepath}: {e}")
            return None
    
    # Assuming your API expects a separate field for nicknames, you might need to adjust this part.
    # If nicknames are expected as part of the files tuple, adjust accordingly.
    data = {'nicknames': ','.join([fn.split(':')[1] for fn in filepaths_with_nicknames])}
    
    response = requests.post(url, files=files, data=data)
    
    # Close the file handles
    for _, file_tuple in files:
        _, file_handle = file_tuple
        file_handle.close()
    
    return response

def delete_files(nicknames):
    # URL of the FastAPI endpoint that handles the DELETE request
    url = "http://127.0.0.1:8000/delete_files/"
    
    # Construct the JSON body with the list of nicknames to be deleted
    data = {"ids": nicknames}
    
    # Send the DELETE request with the JSON body
    response = requests.delete(url, json=data)
    return response


def list_files():
    url = "http://127.0.0.1:8000/list_files/"
    response = requests.get(url)
    return response

def help():
    print("Welcome to the interactive client for FastAPI operations.")
    print("Use /chat <query> to generate a response.")
    print("Use /upload <filepath1>:<nickname1> <filepath2>:<nickname2> ... to upload multiple files with their nicknames.")
    print("Use /delete <nickname1> <nickname2> ... to delete files by nicknames.")
    print("Use /list to list all files.")
    print("Use /help to see the commands again.")
    print("Type /quit to exit the program.")

def main():
    help()
    while True:
        user_input = input("> ")
        if user_input.startswith("/quit"):
            print("Exiting program.")
            break
        elif user_input.startswith("/help"):
            help()
        elif user_input.startswith("/chat"):
            _, prompt = user_input.split(" ", 1)
            response = generate_response(prompt)
            if response.status_code == 200:
                print(response.json())
            else:
                print("Error:", response.status_code, response.text)
        elif user_input.startswith("/upload"):
            _, *filepaths_with_nicknames = user_input.split(" ", 1)
            try:
                filepaths_with_nicknames = [filepaths_with_nicknames[0]]  # Adjust based on how input is split
                response = upload_files(filepaths_with_nicknames)
                if response and response.status_code == 200:
                    print("Success:", response.json())
                else:
                    print("Error:", response.status_code, response.text)
            except Exception as e:
                print("Error:", str(e))
        elif user_input.startswith("/delete"):
            _, *nicknames = user_input.split(" ")
            # Ensure nicknames are sent as a list
            response = delete_files(nicknames)
            if response.status_code == 200:
                print("Success:", response.json())
            else:
                print("Error:", response.status_code, response.text)
        elif user_input.startswith("/list"):
            response = list_files()
            if response.status_code == 200:
                files_info = response.json().get("files", [])
                if files_info:
                    for file in files_info:
                        print(f"Nickname: {file['nickname']}, Original Name: {file['original_name']}")
                else:
                    print("No files found.")
            else:
                print("Error:", response.status_code, response.text)
        else:
            print("Unrecognized command. Please try again.")

if __name__ == "__main__":
    main()
