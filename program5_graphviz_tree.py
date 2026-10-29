# Program 5: Export the Decision Tree using Graphviz.
# If the Graphviz system executable is unavailable, this program falls back to Matplotlib.
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz, plot_tree
from common import load_data, encode_data, train_model

PlayTennis = load_data()
PlayTennis, encoders = encode_data(PlayTennis)
clf, X, y = train_model(PlayTennis)

dot_exe = shutil.which("dot")
possible_bins = [
    r"C:\Program Files\Graphviz\bin",
    r"C:\Program Files (x86)\Graphviz\bin",
]
for folder in possible_bins:
    if os.path.isfile(os.path.join(folder, "dot.exe")):
        os.environ["PATH"] += os.pathsep + folder
        dot_exe = shutil.which("dot")
        break

try:
    from graphviz import Source
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=list(X.columns),
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        special_characters=True
    )
    if not dot_exe:
        raise FileNotFoundError("Graphviz dot.exe is not on PATH.")
    source = Source(dot_data)
    output_path = source.render(filename="DecisionTree_Graphviz", format="png", cleanup=True)
    print(f"Decision tree saved as {output_path}")
except Exception as error:
    print(f"Graphviz export unavailable: {error}")
    print("Creating the tree with Matplotlib instead.")
    plt.figure(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=X.columns,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True
    )
    plt.tight_layout()
    plt.savefig("DecisionTree_Matplotlib.png", dpi=200)
    print("Decision tree saved as DecisionTree_Matplotlib.png")
    plt.show()
