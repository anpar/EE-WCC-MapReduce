import sys

from termcolor import colored

# Print iterations progress
# (improved from https://stackoverflow.com/a/34325723/4103429)
def pprogress(iteration, total, prefix='', suffix='', decimals=1,
              bar_length=100):
    """
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals
        in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))

    if iteration <= 0.33*total:
        color = 'red'
    elif iteration <= 0.66*total:
        color = 'yellow'
    else:
        color = 'green'

    if iteration == total:
        bar = colored('▪' * (filled_length + 1) + '-' * (bar_length - \
                filled_length), color)
    else:
        bar = colored('▪' * filled_length + '▸' + '-' * (bar_length - \
                filled_length), color)


    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar,
                                            percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
