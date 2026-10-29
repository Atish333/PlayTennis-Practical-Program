# Play Tennis Decision Tree Practical (Programs 1–6)

This repository contains the `PlayTennis.csv` dataset and six Python practical programs.
It is designed for Windows and uses paths relative to this folder, so it does not depend on
a particular username or Desktop location.

## Files
- `PlayTennis.csv` — 14-row dataset used by all programs.
- `all_programs.py` — one file containing Programs 1–6. Run this to see the menu.
- `program1_load_csv.py` — read and display the CSV.
- `program2_label_encode.py` — convert text categories to numbers.
- `program3_train_tree.py` — train a Decision Tree using entropy.
- `program4_plot_tree.py` — display/save a tree using Matplotlib.
- `program5_graphviz_tree.py` — export the tree using Graphviz, with a Matplotlib fallback.
- `program6_predict.py` — predict the training rows and compare with the known labels.
- `requirements.txt` — Python package list.

## Setup on a lab computer

Install Python 3.10 or newer from:
https://www.python.org/downloads/windows/

During Python installation, tick **Add python.exe to PATH** if that option appears.

Open Command Prompt in this repository folder and run:

```bat
py -m pip install -r requirements.txt
```

Programs 1–4 and 6 use Python packages installed by that command.
Program 5 can draw a tree using Matplotlib even if the Graphviz executable is not installed.

### Optional: install Graphviz for Program 5's Graphviz export
1. Open the official download page: https://graphviz.org/download/
2. Under Windows, download the 64-bit EXE installer (usually the recommended option).
3. Install it. If the installer offers an option to add Graphviz to PATH, enable it.
4. Close and reopen Command Prompt. Test with:

```bat
dot -V
```

If `dot -V` prints a version, Graphviz is available. The Python package `graphviz` is installed by `requirements.txt`.

## Run the programs

Open Command Prompt in this folder, then run one of these:

```bat
py program1_load_csv.py
py program2_label_encode.py
py program3_train_tree.py
py program4_plot_tree.py
py program5_graphviz_tree.py
py program6_predict.py
```

Or run the combined file:

```bat
py all_programs.py
```

It shows a menu for Programs 1–6. Choose a number.

## Important notes
- Keep `PlayTennis.csv` in the same folder as the Python files.
- Run commands from this folder, or the scripts will tell you where to put the CSV.
- The tree and prediction programs use the same standard 14-row Play Tennis dataset.
- Program 6 compares predictions against the same rows used to train the model. This is a practical demonstration, not a test of accuracy on unseen data.
