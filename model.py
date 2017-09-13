import os
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Input, Lambda
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

BATCH_SIZE=32
EPOCHS = 150
CENTER = "center"
LEFT = "left"
RIGHT = "right"
STEERING = "steering"
cam_corrections = { CENTER : 0, LEFT : 0.2, RIGHT : -0.2}

def image_flip(image, angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle

def generator(df, is_training, batch_size=BATCH_SIZE):
    num_samples = df.shape[0] #or df['center'].count()
    while 1:
        shuffle(df) # or df.sample(frac=1).reset_index(drop=True)
        for offset in range(0, num_samples, batch_size):
            df_batch = df.iloc[offset:offset + batch_size] #slice wont raise out of bound exception
            images = []
            angles = []
            for index, row in df_batch.iterrows():
                for pos in cam_corrections:
                    image = cv2.imread(os.path.join('./', row[pos].strip()))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = row[STEERING] + cam_corrections[pos]
                    if is_training:
                        image, angle = image_flip(image, angle)
                    
                    images.append(image)
                    angles.append(angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

def get_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((40, 40), (0, 0)), input_shape=(160, 320, 3)))
    model.add(BatchNormalization())
    #cov2d 1
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(0.2))
    #conv2d 2
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))
    #conv2d 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))
    #flatten
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    return model

def train(model, df_train, df_val):
    callbacks = [EarlyStopping(monitor='val_loss', patience=30, verbose=0),
                 ModelCheckpoint('model.hdf5', monitor='val_loss', save_best_only=True, verbose=0)]
    adam_opt=Adam(lr=5e-5)
    model.compile(adam_opt, 'mse', ['accuracy'])
    model_history = model.fit_generator(generator(df_train, False), epochs=EPOCHS,
                                        steps_per_epoch=int(df_train.shape[0]/BATCH_SIZE),
                                        validation_data=generator(df_val, True),
                                        validation_steps=int(df_val.shape[0]/BATCH_SIZE),
                                        callbacks = callbacks)
    return model_history

def main():
    df_collect_0 = pd.read_csv('./data_collect_0/driving_log.csv')
    df_collect_1 = pd.read_csv('./data_collect_1/driving_log.csv')
    df_collect_2 = pd.read_csv('./data_collect_2/driving_log.csv')
    frames = [df_collect_0, df_collect_1, df_collect_2]
    df = pd.concat(frames)
    df_train, df_val = train_test_split(df, test_size=0.2)

    model = get_model()
    model_history = train(model, df_train, df_val)
    with open('model_history.p','wb') as f:
        pickle.dump(model_history.history, f)
    
if __name__=='__main__':
    main()
