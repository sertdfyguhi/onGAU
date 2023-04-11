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


def resize_size_to_fit(
    image_size: tuple[int, int] | list[int, int],
    window_size: tuple[int, int] | list[int, int],
):
    """
    Recursive function to resize a specified image size to fit into a specified window size.
    """

    result_image_size = []

    if (is_width := image_size[0] > window_size[0]) or (image_size[1] > window_size[1]):
        aspect_ratio = image_size[0] / image_size[1]

        if is_width:
            result_image_size = [window_size[0], round(window_size[0] / aspect_ratio)]
        else:
            result_image_size = [round(window_size[1] * aspect_ratio), window_size[1]]
    else:
        return image_size

    return resize_size_to_fit(result_image_size, window_size)


def append_dir_if_startswith(path: str, dir: str, startswith: str):
    """Checks if a path starts with and if so appends a path to it"""
    if path.startswith(startswith):
        return os.path.join(dir, path)
    return path
