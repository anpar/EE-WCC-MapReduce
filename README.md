# Requirements
The code available in this repository was tested with:
- Python 3.6.7
- NumPy 1.15.4
- SciPy 1.1.0
- matplotlib 3.0.2
- termcolor 1.1.0 (this one is not strictly useful, it only enhance the readbility of the debug information in the terminal and shows nice colored progress bar while the simulation is running).

# Organization
The entire source code is located in ``src/``.

The file ``core.py`` contains the "logic" needed (e.g. Algorithm 1 in the paper).
The file ``utils.py`` contains a small script displaying the progress of a numerical experiments in the terminal while running.
The files ``figure2.py``, ``figures3ab.py`` and ``figure3c.py`` allow to reproduce the figures given in the paper.

# Run
To generate Figure 2, run the command 
```
python3 figure2.py
```
Not that this takes some time.

To generate Figures 3a and 3b, run the command
```
python3 -O figures3ab.py
```
(without the ``-O`` flag if you want debug information to appear in your terminal). Note that this takes some time.

To generate Figure 3c, run the command
```
python3 -O figure3c.py
```
(without the ``-O`` flag if you want debug information to appear in your terminal). Note that this takes some time.

# Copyright and license
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

# Acknowledgment
Antoine Paris is a Research Fellow of the F.R.S.-FNRS. This work was also
supported by F.R.S.-FNRS under the EOS program (project 30452698,
“MUlti-SErvice WIreless NETwork”).

# Contact
For feedback, comments, bug reports, etc, please contact antoine.paris [at] uclouvain.be.
