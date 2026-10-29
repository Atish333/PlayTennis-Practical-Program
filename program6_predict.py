# Program 6: Predict the Play Tennis labels and compare with the known labels
from common import load_data, encode_data, train_model

PlayTennis = load_data()
PlayTennis, encoders = encode_data(PlayTennis)
clf, X, y = train_model(PlayTennis)

X_pred = clf.predict(X)
print("Predicted values:")
print(X_pred)
print("\nPrediction matches actual label:")
print(X_pred == y.to_numpy())
