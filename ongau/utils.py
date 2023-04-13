import os


# edited and modified from https://stackoverflow.com/questions/17984809/how-do-i-create-an-incrementing-filename-in-python
def next_file_number(path_pattern: str):
    """
    Finds the next free file number in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    1
    2
    3

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return b


def resize_size_to_fit(image_size, window_size):
    """
    Function to resize a specified image size to fit into a specified window size.
    AI generated code that is faster than my previous implementation.
    """

    image_width, image_height = image_size
    window_width, window_height = window_size

    image_ratio = image_width / image_height
    window_ratio = window_width / window_height

    if image_ratio > window_ratio:
        # Scale the image to fit within the window height and center it horizontally
        scale_factor = window_height / image_height
        scaled_width = int(image_width * scale_factor)

        if scaled_width <= window_width:
            return (scaled_width, window_height)
        else:
            scaled_width = window_width
            scaled_height = int(image_height * scaled_width / image_width)
            return (scaled_width, scaled_height)
    else:
        # Scale the image to fit within the window width and center it vertically
        scale_factor = window_width / image_width
        scaled_height = int(image_height * scale_factor)

        if scaled_height <= window_height:
            return (window_width, scaled_height)
        else:
            scaled_height = window_height
            scaled_width = int(image_width * scaled_height / image_height)
            return (scaled_width, scaled_height)


def append_dir_if_startswith(path: str, dir: str, startswith: str):
    """Checks if a path starts with and if so appends a path to it."""
    return os.path.join(dir, path) if path.startswith(startswith) else path
