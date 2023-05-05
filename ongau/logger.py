# felt like adding colors to terminal output
RESET = "\033[m"
BOLD = "1"
ERROR = "31"
SUCCESS = "32"
WARNING = "33"
INFO = "36"


def create(msg: str, codes: list[int]):
    return f"\033[{';'.join(codes)}m{msg}{RESET}"


def error(msg: str):
    print(f"{create('ERROR: ', [ERROR, BOLD])}{create(msg, [ERROR])}")


def warn(msg: str):
    print(f"{create('WARNING: ', [WARNING, BOLD])}{create(msg, [WARNING])}")


def success(msg: str):
    print(create(msg, [SUCCESS]))


def info(msg: str):
    print(create(msg, [INFO]))
