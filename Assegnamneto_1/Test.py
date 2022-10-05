"""First assignment for the CMEPDA course, 2022/23.
"""

import argparse #For all the stuff done by command line
import string #For the function string.ascii
import numpy #For array of letters count
import time #For elapsed time
from matplotlib import pyplot

#Global variables
start1 = time.time()#The beginning of the time
N = 26 #Numbers of letters

#Call the function that counts letters from the point you want to start from the point you want to end the reading. This function also prints an instogram done by * and another done by mathplotlib
def parsefile(input_file, start, finish, histo):

    #Create a alpha-numeric dictionary for letters search
    alphabet = string.ascii_lowercase
    numbers = range(N)

    dict1 = dict(zip(alphabet, numbers))
    #dict2 = dict(zip(Alphabet, numbers))

    #With DICT  decide whether a word is upper or lower
    DICT = dict1 #or dict2

    #Get rid of useless part in the text with an else if. In case of no necessity of cuts, the frograms strarts from the beginning and arrive at the end
    #i = input_file.find('*** START OF THE PROJECT GUTENBERG EBOOK THE REPUBLIC ***'.lower()) #Useful with 'CHAPTER 1' or 'PART 1'
    #j = input_file.find('*** END OF THE PROJECT GUTENBERG EBOOK THE REPUBLIC ***'.lower()) #Useful with 'THE END' or 'The End'
    if start != '':
        i = input_file.find(start.lower()) #Useful with 'CHAPTER 1' or 'PART 1'
    else:
        i = 0
    if finish != '':
        j = input_file.find(finish.lower()) #Useful with 'THE END' or 'The End'
    else:
        j = len(input_file)
    input_file = input_file[i:j]
    #print(input_file)

    #Letters detection, histogram bins construction and features counting
    bins = numpy.zeros(len(alphabet))

    n = 0 #Index for values in the dictionary
    l = 0 #Counter of letters met
    c = 0 #Counter of total characters
    w = 0 #Number of word
    p = 0 #Control parameter

    lines = 0 #Lines counter
    for x in input_file:
        c = c + 1
        if x in DICT:
            l = l + 1
            n = DICT.get(x)
            bins[n] = bins[n] + 1 #Whitout square braket?
        if x == ' ' or x =='\n':
            if p == 0:
                w = w + 1
                p = 1
        elif x != ' ':
            p = 0
        if x=='\n':
            lines = lines + 1

    #In the end, count the bins
    bins = bins / l * 100

    #A funny way to print the results of the exercise
    print(f'Percentage content of the bins: \n')
    for i in range(N):
        car = alphabet[i]
        print(car, "{0:6.2f}".format(bins[i])+"%",  '*'*int(5*bins[i]))

    #In case there's no need to print the istogram
    if not histo:
        return

    #Print some informations
    print(f'Number of letters: {l}')
    print(f'Number of characters: {c}')
    print(f'Number of words: {w}')
    print(f'Number of lines: {lines}')

    #Histogram (to be fixed)
    fig, ax = pyplot.subplots(1, 1)
    ax.hist(range(N), len(alphabet), weights = bins)

    #Next time I'll make a function for the istorams...
    ax.set_title("Histogram of letters appearances")
    ax.set_xlabel('Letters')
    ax.set_ylabel('Percentage')

    #Giorgione... Please explain me what is this :)
    rects = ax.patches
    labels = alphabet
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
                ha='center', va='bottom')

    end = time.time()
    print(f'Total elapsed time: {end - start1}')

    #This pyplot is very annoying... I found no way to cut it off when the program is running
    #pyplot.show()


def process(file_path, start, finish, histo):
    """
    """
    #print(f'Opening input file {file_path}...')

    with open(file_path, 'r') as input_file:
        text = input_file.read().lower()
    parsefile(text, start, finish, histo)

    #print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print some book statistics')
    parser.add_argument('infile', type=str, help='Path to the input file')
    parser.add_argument('--start', type=str, default='', help='Start of the document')
    parser.add_argument('--end', type=str, default='',  help='End of the document')
    parser.add_argument('--histo', action='store_true', help='Show a letter histogram')
    args = parser.parse_args()
    process(args.infile, args.start, args.end, args.histo)