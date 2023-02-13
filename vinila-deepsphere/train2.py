import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')
batch_size=32
epochs=40
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:3","/gpu:2","/gpu:1","/gpu:0"])
class LearningRateReducerCb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.98
    self.model.optimizer.lr.assign(new_lr)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

def draw_performance(x,y,title):  
  plt.clf()
  plt.figure(figsize=(12,8))
  plt.grid()
  plt.scatter(x,y, label="predictions", color='red')
  mi = np.amin(x)
  ma = np.amax(x)
  plt.plot([np.amin(x), np.amax(x)], [np.amin(x), np.amax(x)], 'k-', alpha=0.75, zorder=0, label="y=x")
  plt.legend()
  plt.xlabel("True")
  plt.ylabel("Predicted")
  plt.title(title)
  plt.savefig("./result4/%s.png" %title)


theta = np.load("y_p.npy")[:,0]
#phi = np.load("/home/mengty/temp/y_p.npy")[:,1]
#x = np.sin(theta)*np.cos(phi)
#y=  np.sin(theta)*np.sin(phi)
theta = theta[:,np.newaxis]
y_all = np.concatenate((np.sin(theta),np.cos(theta)),axis=1)


#y_all1=np.sin(y_all)
#y_all=np.cos(y_all)
#y_all=np.concatenate((y_all1[:,np.newaxis],y_all2[:,np.newaxis]),axis=1)
x_slope=np.load("slope_p.npy")
x_npe=np.load("npe_p.npy")
#x_mediantime=np.load("peaktime_7.npy")#[1:100,:,:]
x_fht=np.load("fht_p.npy")
x_peak = np.load("peak_p.npy")
#x_nperatio = np.load("/home/mengty/temp/nperatio_p.npy")
x_slope=x_slope[:,:,:,np.newaxis]
#x_mediantime=x_mediantime[:,:,:,np.newaxis]
x_npe=x_npe[:,:,:,np.newaxis]
x_fht=x_fht[:,:,:,np.newaxis]
x_peak=x_peak[:,:,:,np.newaxis]
#x_nperatio = x_nperatio[:,:,:,np.newaxis]
x_all=np.concatenate((x_slope,x_npe,x_fht,x_peak),axis=3)
del x_fht
del x_npe
del x_slope
del x_peak
x_all[x_all==1250]=0
#x_all = x_all[energy>1.0]

print("y_shape={},x_shape={}".format(y_all.shape,x_all.shape))
input_shape=(None,126,252,4)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
del x_all
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#train_dataset = train_dataset.shuffle(100).batch(batch_size)
#test_dataset = test_dataset.batch(batch_size)

#feature = tf.keras.applications.resnet50.ResNet50(include_top=False,weights=None,input_shape=input_shape[1:],classes=2)
with strategy.scope():
    #feature = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,weights=None,input_shape=input_shape[1:],classes=1)
    feature= tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,weights=None,input_shape=input_shape[1:])
    model = tf.keras.Sequential([feature,
                                # tf.keras.layers.Conv2D(1280,(1,4),activation=None),
                                 tf.keras.layers.BatchNormalization(axis=3),
                                 tf.keras.layers.ReLU(),
                                 tf.keras.layers.Dropout(rate=0.2),
                               #  tf.keras.layers.Conv2D(,(3,3),activation=None),
                                # tf.keras.layers.GlobalAvgPool2D(),
                               #  tf.keras.layers.BatchNormalization(axis=3),
                               #  tf.keras.layers.ReLU(), 
                                # tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.2),
                                 #tf.keras.activations.swish(),
                                 #tf.keras.layers.BatchNormalization(axis=3),
                                 tf.keras.layers.Dense(256,activation="relu"),
                                 tf.keras.layers.Dense(64,activation = 'relu'),
                                 tf.keras.layers.Dense(2)
                                ])

    #model.build(input_shape=input_shape)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(lr=0.0015),
              metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary(110)
    history = model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test,y_test),
          callbacks=[LearningRateReducerCb(),es_callback])
    model.save_weights('./result4/weights_5features_sPMT.h5')

    plt.figure(figsize=(12,8))
    plt.plot(history.history["loss"], label="training")
    plt.plot(history.history["val_loss"], label="validation")
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.savefig("learning_curve_theta.png")
print(model.evaluate(x_train,y_train))
print(model.evaluate(x_test,y_test))

#y_train_true = np.concatenate([y for x, y in train_dataset], axis=0)
#print(y_train_true.shape)
predictions = model.predict(x_train)
#predictions=np.arccos(predictions)
#y_train=np.arccos(y_train)
draw_performance(np.arctan(y_train[:,0]/y_train[:,1]),np.arctan(predictions[:,0]/predictions[:,1]),'train_prediction_theta')
np.save("./result4/theta_predict_train.npy", predictions)
np.save("./result4/theta_true_train.npy", y_train)

#y_test_true = np.concatenate([y for x, y in test_dataset], axis=0)
#print(y_test_true.shape)
predictions = model.predict(x_test)
#predictions=np.arccos(predictions)
#y_test=np.arccos(y_test[:,0])
draw_performance(np.arctan(y_test[:,0]/y_test[:,1]),np.arctan(predictions[:,0]/predictions[:,1]), 'prediction_theta')
np.save("./result4/theta_predict.npy", predictions)
np.save("./result4/theta_true.npy",y_test)
  #draw_hists(y_test, predictions)





