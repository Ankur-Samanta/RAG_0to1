import requests

def generate_response(prompt):
    """
    Sends a POST request to generate a response for a given prompt.

    Parameters:
    - prompt (str): The input prompt for which the response is generated.

    Returns:
    - response (Response): The response object from the requests library.
    """
    url = "http://127.0.0.1:8000/generate_response"
    params = {"prompt": prompt}
    response = requests.post(url, params=params)
    return response

def upload_files(filepaths_with_nicknames):
    """
    Uploads files to the server with their respective nicknames.

    Parameters:
    - filepaths_with_nicknames (list of str): Each element is a string in the format "filepath:nickname".

    Returns:
    - response (Response): The response object from the requests library, or None if an error occurs.
    """
    url = "http://127.0.0.1:8000/upload_files/"
    files = []
    for file_nickname_pair in filepaths_with_nicknames:
        filepath, nickname = file_nickname_pair.split(':')
        try:
            file_handle = open(filepath, 'rb')
            files.append(('files', (nickname, file_handle)))
        except Exception as e:
            print(f"Error opening {filepath}: {e}")
            return None
    
    data = {'nicknames': ','.join([fn.split(':')[1] for fn in filepaths_with_nicknames])}
    response = requests.post(url, files=files, data=data)
    for _, file_tuple in files:
        _, file_handle = file_tuple
        file_handle.close()
    return response

def delete_files(nicknames):
    """
    Deletes files by their nicknames.

    Parameters:
    - nicknames (list of str): The nicknames of the files to delete.

    Returns:
    - response (Response): The response object from the requests library.
    """
    url = "http://127.0.0.1:8000/delete_files/"
    data = {"ids": nicknames}
    response = requests.delete(url, json=data)
    return response

def list_files():
    """
    Retrieves a list of all files.

    Returns:
    - response (Response): The response object from the requests library.
    """
    url = "http://127.0.0.1:8000/list_files/"
    response = requests.get(url)
    return response

def help():
    """
    Prints the help menu with available commands.
    """
    print("Welcome to the interactive client for FastAPI operations.")
    print("Use /chat <query> to generate a response.")
    print("Use /upload <filepath1>:<nickname1> <filepath2>:<nickname2> ... to upload files with nicknames.")
    print("Use /delete <nickname1> <nickname2> ... to delete files by nicknames.")
    print("Use /list to list all files.")
    print("Use /help to see this menu again.")
    print("Type /quit to exit the program.")

def main():
    """
    Main function to run the interactive client.
    """
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
