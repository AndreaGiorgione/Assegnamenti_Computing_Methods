# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.

"""First assignment for the CMEPDA course, 2022/23.
"""

import argparse # For entering arguments in functions
import time # For elapsed time
import string # For the function string.ascii
import numpy # For array of letters count


# Global variables
T_0 = time.time() # The beginning of the time
N = 26 # Numbers of letters


def parsefile(text, start, finish, histo, stats, helping):
    '''Features counter function and processer of
    information in text.
    Input:
        a string of caracthers.
    Parameters:
        start (string): The starting string of the analysis.
        finish (string): The first string not included in the analysis.
        histo (action): The signal to plot the frequency letter histogram.
        stats (action): The signal to print geeral stats of the file.
        helping (action): the signal user exploit to recive an help if needed.
    Returns:
        print of letter frequencies, number of features, plot of frequencies histogram
    '''

    # Give help if nedded
    if helping:
        print("""
        Use --start to indicate the starting line of the file analysis (first line counted).
        Use --end to indicate the ending line of the file analysis (first line not counted).
        Use --histo to plot an histogram.
        Use --stats to print some basic stats of the file.
        Use --helping to get help.
        """)
        return

    # Create a alpha-numeric dictionary for letters search
    alphabet = string.ascii_lowercase
    numbers = range(N)

    dictionary = dict(zip(alphabet, numbers))

    # Get rid of useless text part if needed
    if start != '':
        i = text.find(start.lower()) # Useful with 'CHAPTER 1' or 'PART 1'
    else:
        i = 0
    if finish != '':
        j = text.find(finish.lower()) # Useful with 'THE END' or 'The End'
    else:
        j = len(text)
    text = text[i:j]

    # Letters detection, histogram bins construction and features counting
    bins = numpy.zeros(len(alphabet))

    i = 0 # Index for values in the dictionary
    parameter = 0 # Control parameter (for counting words)
    lett = 0 # Counter of letters met
    char = 0 # Counter of total characters
    words = 0 # Number of words

    lines = 0 # Lines counter
    for element in text:
        char = char + 1
        if element in dictionary:
            lett = lett + 1
            i = dictionary.get(element)
            bins[i] = bins[i] + 1
        if element in (' ', '\n'):
            if parameter == 0:
                words = words + 1
                parameter = 1
        elif element != ' ':
            parameter = 0
        if element == '\n':
            lines = lines + 1

    # Count the bins and print the results
    bins = bins / lett * 100
    print(f'Results: \n {dict(zip(alphabet, bins))}')

    # Print some informations
    if stats:
        print(f'''
        Number of letters: {lett}
        Number of characters: {char}
        Number of words: {words}
        Number of lines: {lines} \n
        ''')

    # In case there's no need to print the istogram
    if not histo:
        # Count the time in case of no istogram
        end = time.time()
        print(f'Total elapsed time: {end - T_0}')
        return

    # A funny way to print the histogram
    print('Percentage content of the bins:')
    for i in range(N):
        car = alphabet[i]
        print(car, "{0:6.2f}".format(bins[i]) + "%",  '*'*int(5*bins[i]))

    # Count the time in case of Istogram
    end = time.time()
    print(f'Total elapsed time: {end - T_0}')
    return


def process(file_path, start, finish, histo, stats, helping):
    """Opening of a file path-name in to a string carrying
    some arguments for next implementation.
    Input:
        path-name of the desired file.
    Parameters:
        start (string): The starting string of the analysis.
        finish (string): The first string not included in the analysis.
        histo (action): The signal to plot the frequency letter histogram.
        stats (action): The signal to print geeral stats of the file.
        helping (action): the signal user exploit to recive an help if needed.
    Returns:
        the input file open in string form
    """
    print(f'Opening input file {file_path}...')
    with open(file_path, 'r', encoding="utf-8") as text:
        text = text.read().lower()
    parsefile(text, start, finish, histo, stats, helping)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print some book statistics')
    parser.add_argument('infile', type=str, help='Path to the input file')
    parser.add_argument('--start', type=str, default='', help='Start of the document')
    parser.add_argument('--end', type=str, default='',  help='End of the document')
    parser.add_argument('--histo', action='store_true', help='Show a letter histogram')
    parser.add_argument('--stats', action='store_true', help='Show general stats of the book')
    parser.add_argument('--helping', action='store_true', help='Give an help to user')
    args = parser.parse_args()

    process(args.infile, args.start, args.end, args.histo, args.stats, args.helping)