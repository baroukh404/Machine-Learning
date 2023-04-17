from gekko import brain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# donne d'apprentissage
x = pd.read_csv("train.csv")
x = x.drop('price_range', axis=1)
x = x.to_numpy()
x = x.reshape(x.shape[-1], x.shape[0])
x = x.tolist()
y = pd.read_csv("train.csv")  
y = y['price_range']
y = y.to_numpy()
y = y.reshape(1, y.shape[0])

# neral network
b = brain.Brain(remote=False)
b.input_layer(20)
b.layer(sigmoid=40)
b.layer(sigmoid=40)
b.layer(sigmoid=40)
b.layer(sigmoid=40)
b.layer(sigmoid=40)
b.output_layer(1)
b.learn(x, y)

# testing our network
xp = pd.read_csv('test.csv')
xp = xp.drop('id', axis=1)
xp = xp.to_numpy()
xp = xp.reshape(xp.shape[-1], xp.shape[0])
yp = b.think(xp)
prediction = {
	'price_range': yp.tolist()
}
prediction = pd.DataFrame(prediction)
prediction.to_csv('backup_price.csv', index=False)

# plt.figure()
# plt.scatter(x, y)
# plt.scatter(x, yp)
# plt.show()