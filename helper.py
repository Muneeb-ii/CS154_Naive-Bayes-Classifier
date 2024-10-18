def get_file_contents(filename):
    """
    Reads the contents of a file and returns it as a string.
    
    Parameters:
        filename (str): The name of the file to read.
    Returns:
        str: The contents of the file as a string, or None if an error occurs.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            contents = file.read()
        return contents
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except IOError:
        print(f"Error: An I/O error occurred while reading '{filename}'.")
        return None
