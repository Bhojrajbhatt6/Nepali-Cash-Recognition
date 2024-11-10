import tensorflow as tf
model = tf.keras.models.load_model('cashRec.h5')

conv_layers = 0
dense_layers = 0

for layer in model.layers:
    if 'conv' in layer.name:
        conv_layers += 1
    elif 'dense' in layer.name:
        dense_layers += 1

# print('Number of convolutional layers:', conv_layers)
# print('Number of dense layers:', dense_layers)


# for layer in model.layers:
#     print(layer.name, layer.__class__.__name__)

print(model.summary())