# Program 4: Visualize the Decision Tree with Matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from common import load_data, encode_data, train_model

PlayTennis = load_data()
PlayTennis, encoders = encode_data(PlayTennis)
clf, X, y = train_model(PlayTennis)

plt.figure(figsize=(14, 8))
tree.plot_tree(
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
