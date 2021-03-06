"""
MIT License

Copyright (c) 2019 Université Catholique de Louvain (UCLouvain)

The software provided allows to reproduce the results presented in the
research paper "Energy-Efficient Edge-Facilitated Wireless Collaborative
Computing using Map-Reduce" by Antoine Paris, Hamed Mirghasemi, Ivan Stupia
and Luc Vandendorpe from ICTEAM/ELEN/CoSy (UCLouvain).

Contact: antoine.paris@uclouvain.be

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
