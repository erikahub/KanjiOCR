"""sources: 
https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
https://stackoverflow.com/questions/17771287/python-octal-escape-character-033-from-a-dictionary-value-translates-in-a-prin/17772337"""

class bcolors:
    """Use as such:
    print(f"{bcolors.FAIL}Warning: This is a warning message in colour. Continue?{bcolors.ENDC}")
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# print(f"{bcolors.WARNING}Warning: This is a warning message in colour. Continue?{bcolors.ENDC}")

# color = {
#     'white':    "\033[1;37m",
#     'yellow':   "\033[1;33m",
#     'green':    "\033[1;32m",
#     'blue':     "\033[1;34m",
#     'cyan':     "\033[1;36m",
#     'red':      "\033[1;31m",
#     'magenta':  "\033[1;35m",
#     'black':      "\033[1;30m",
#     'darkwhite':  "\033[0;37m",
#     'darkyellow': "\033[0;33m",
#     'darkgreen':  "\033[0;32m",
#     'darkblue':   "\033[0;34m",
#     'darkcyan':   "\033[0;36m",
#     'darkred':    "\033[0;31m",
#     'darkmagenta':"\033[0;35m",
#     'darkblack':  "\033[0;30m",
#     'off':        "\033[0;0m"
# }