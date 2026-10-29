# Program 3: Train a Decision Tree using entropy
from common import load_data, encode_data, train_model

PlayTennis = load_data()
PlayTennis, encoders = encode_data(PlayTennis)
clf, X, y = train_model(PlayTennis)
print(clf)
