# Requirements
The code available in this repository was tested with:
- Python 3.6.7
- NumPy 1.15.4
- SciPy 1.1.0
- matplotlib 3.0.2
- termcolor 1.1.0

# Files
The entire source code is located in ``src/``.

The file ``core.py`` contains the "logic" needed (e.g. Algorithm 1 in the paper).
The file ``utils.py`` contains a small script displaying the progress of a numerical experiments in the terminal while running.
The files ``figure2.py``, ``figures3and4.py`` and ``figure5.py`` allow to reproduce the figures given in the paper.

# Run
To generate Figure 2, run the command 
```
python3 figure2.py
```

To generate Figures 3a and 3b, run the command
```
python3 -O figures3ab.py
```
(without the ``-O`` flag if you want debug information to appear in your terminal).

To generate Figure 3c, run the command
```
python3 -O figure3c.py
```
(without the ``-O`` flag if you want debug information to appear in your terminal).

# Contact
For feedback, comments, bug reports, etc, please contact antoine.paris [at] uclouvain.be.
