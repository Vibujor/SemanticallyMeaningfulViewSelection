"""Script for training the view selection model."""

import h5py

from keras.models import Model
from keras.layers import Dense, Activation, Input, concatenate, Dropout
from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam


#extract training data and testing data
f = h5py.File("vgg_block5pool_trainset.h5", "r")
train_features = f["features"][:]
train_th_ph = f["th_ph"][:]
train_labels = f["labels"][:]
f.close()

f = h5py.File("vgg_block5pool_testset.h5", "r")
test_features = f["features"][:]
test_th_ph = f["th_ph"][:]
test_labels = f["labels"][:]
f.close()

#input layer
#MLP1
im_input = Input(shape=(25088, ))
im_dense = Dense(4096, )(im_input)
im_norm = BatchNormalization()(im_dense)
im_activation = Activation("relu")(im_norm)

drop1 = Dropout(0.75)(im_norm)
im_dense2 = Dense(4096, )(drop1)
im_norm2 = BatchNormalization()(im_dense2)
im_activation2 = Activation("relu")(im_norm2)

#concatenate with angular inputs
ang_input = Input(shape=(2, ))

merged = concatenate([im_activation2, ang_input])

#MLP2
drop2 = Dropout(0.75)(merged)
dense2 = Dense(4096, )(drop2)
norm2 = BatchNormalization()(dense2)
activation2 = Activation("relu")(norm2)

drop3 = Dropout(0.75)(activation2)
dense3 = Dense(4096, )(drop3)
norm3 = BatchNormalization()(dense3)
act3 = Activation("relu")(norm3)

drop = Dropout(0.75)(act3)
dense_final = Dense(1, )(drop)
norm_final = BatchNormalization()(dense_final)
act_final = Activation("sigmoid")(norm_final)

adam = Adam(lr=1e-3)

model = Model(inputs=[im_input, ang_input], outputs=act_final)
model.compile(optimizer=adam, loss='mse')

try:
    model.load_weights("my_weights.h5")
except Exception:
    pass

#train model on 10 epochs
model.fit([train_features, train_th_ph], train_labels,
          shuffle=True, epochs=10, batch_size=32,
          validation_data=([test_features, test_th_ph], test_labels))

# results
print(model.predict([test_features, test_th_ph]))

#save model as json and weights as HDF5
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("my_weights.h5")
