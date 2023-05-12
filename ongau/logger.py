# Fancy logging with colors.

# ANSI escape codes
RESET = "\033[m"
BOLD = "1"
ERROR = "31"
SUCCESS = "32"
WARNING = "33"
INFO = "36"


def create(msg: str, codes: list[int]):
    """Creates log message using the specified message and ANSI escape codes."""
    return f"\033[{';'.join(codes)}m{msg}{RESET}"


def error(msg: str):
    """Logs an error message."""
    print(f"{create('ERROR: ', [ERROR, BOLD])}{create(msg, [ERROR])}")


def warn(msg: str):
    """Logs a warning message."""
    print(f"{create('WARNING: ', [WARNING, BOLD])}{create(msg, [WARNING])}")


def success(msg: str):
    """Logs a success message."""
    print(create(msg, [SUCCESS]))


def info(msg: str):
    """Logs an info message."""
    print(create(msg, [INFO]))
