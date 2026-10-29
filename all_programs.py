# Combined file: Programs 1–6
from pathlib import Path
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "PlayTennis.csv"

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Keep PlayTennis.csv in this folder: {BASE_DIR}")
    return pd.read_csv(CSV_PATH)

def encode_data(data):
    encoded = data.copy()
    encoders = {}
    for column in encoded.columns:
        encoders[column] = LabelEncoder()
        encoded[column] = encoders[column].fit_transform(encoded[column].astype(str))
    return encoded, encoders

def train(encoded):
    y = encoded["Play Tennis"]
    X = encoded.drop(columns=["Play Tennis"])
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
    clf.fit(X, y)
    return clf, X, y

def program1():
    print("\nPROGRAM 1: Load CSV")
    data = load_data()
    print(data)

def program2():
    print("\nPROGRAM 2: Label Encoding")
    data, _ = encode_data(load_data())
    print(data)

def program3():
    print("\nPROGRAM 3: Train Decision Tree")
    data, _ = encode_data(load_data())
    clf, X, y = train(data)
    print(clf)

def program4():
    print("\nPROGRAM 4: Plot Decision Tree using Matplotlib")
    data, _ = encode_data(load_data())
    clf, X, y = train(data)
    plt.figure(figsize=(14, 8))
    plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"],
              filled=True, rounded=True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "DecisionTree_Matplotlib.png", dpi=200)
    print(f"Saved: {BASE_DIR / 'DecisionTree_Matplotlib.png'}")
    plt.show()

def program5():
    print("\nPROGRAM 5: Graphviz Decision Tree")
    data, _ = encode_data(load_data())
    clf, X, y = train(data)
    for folder in [r"C:\Program Files\Graphviz\bin",
                   r"C:\Program Files (x86)\Graphviz\bin"]:
        if os.path.isfile(os.path.join(folder, "dot.exe")):
            os.environ["PATH"] += os.pathsep + folder
            break
    try:
        from graphviz import Source
        dot_data = export_graphviz(clf, out_file=None, feature_names=list(X.columns),
                                   class_names=["No", "Yes"], filled=True,
                                   rounded=True, special_characters=True)
        if not shutil.which("dot"):
            raise FileNotFoundError("Graphviz dot.exe not found on PATH.")
        output = Source(dot_data).render(
            filename=str(BASE_DIR / "DecisionTree_Graphviz"), format="png", cleanup=True
        )
        print(f"Saved: {output}")
    except Exception as exc:
        print(f"Graphviz unavailable ({exc}). Using Matplotlib fallback.")
        plt.figure(figsize=(14, 8))
        plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"],
                  filled=True, rounded=True)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "DecisionTree_Matplotlib.png", dpi=200)
        print(f"Saved: {BASE_DIR / 'DecisionTree_Matplotlib.png'}")
        plt.show()

def program6():
    print("\nPROGRAM 6: Predict and compare")
    data, _ = encode_data(load_data())
    clf, X, y = train(data)
    pred = clf.predict(X)
    print("Predicted values:")
    print(pred)
    print("\nPredictions match actual labels:")
    print(pred == y.to_numpy())

def main():
    actions = {
        "1": program1, "2": program2, "3": program3,
        "4": program4, "5": program5, "6": program6
    }
    print("Play Tennis Practical: Programs 1–6")
    print("Choose 1, 2, 3, 4, 5, 6 or 0 to exit.")
    while True:
        choice = input("\nEnter program number: ").strip()
        if choice == "0":
            break
        action = actions.get(choice)
        if action is None:
            print("Please enter a number from 1 to 6, or 0 to exit.")
            continue
        try:
            action()
        except Exception as exc:
            print(f"Program could not run: {exc}")
            print("Check README.md for setup instructions.")

if __name__ == "__main__":
    main()
