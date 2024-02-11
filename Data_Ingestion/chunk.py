

def chunk_text(text, chunk_size=500):
    """
    Splits text into chunks of a specified size.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i+chunk_size])