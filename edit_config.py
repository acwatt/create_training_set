#!C:\ProgramData\Anaconda3 python3
import fileinput

def replace_text_in_file(filename, text_to_search, replacement_text):
    with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end='')

