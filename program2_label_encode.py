# Program 2: Convert text categories into numeric labels
from common import load_data, encode_data

PlayTennis = load_data()
PlayTennis, encoders = encode_data(PlayTennis)
print(PlayTennis)
