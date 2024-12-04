# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.src.models import Sequential
from keras.src.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.src.callbacks import ReduceLROnPlateau
from sklearn.decomposition import PCA
import os

# #%%
# Windows
# DATA_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\raw_crowd_train.tsv"
# AUDIO_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\wavs\\"

# MacOS
DATA_TRAIN = "/Users/s.pentin/Yandex.Disk.localized/Учеба/ДИПЛОМ/DataSet/crowd/crowd_train/raw_crowd_train.tsv"
AUDIO_TRAIN = "/Users/s.pentin/Yandex.Disk.localized/Учеба/ДИПЛОМ/DataSet/crowd/crowd_train/wavs/"

# Windows
# DATA_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\raw_crowd_test.tsv"
# AUDIO_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\wavs\\"

EPOCH = 100


def distribution_data(file):
    X = file.iloc[:, :-1].values
    Y = file['labels'].values
    return X, Y


def one_hot_encoder(Y):
    encoder = OneHotEncoder()  # As this is a multiclass classification problem onehotencoding our Y.
    Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    return encoder, Y


# %%
# splitting data
def splitting_data(X, Y, test=0.2):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test, random_state=0, shuffle=True)
    return x_train, x_test, y_train, y_test


# %%
# scaling our data with sklearn's Standard scaler
def standard_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


# %%
# Применение PCA
def pca_for_n_component(x_train, x_test, component=0.9):
    pca = PCA(n_components=component)  # Выбор числа компонент, сохраняющего 90% дисперсии n_components=0.9
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca


# %%
def create_model(x_train):
    model = Sequential()
    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu',
                     input_shape=(x_train.shape[1], 1)))  # 256 -> 128
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    # model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))  # add 256
    # model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    #
    # model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))  # add 512
    # model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=4, activation='softmax'))  # unites - количество эмоций для обучения
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


# %%
def fit_model(model: Sequential, x_train, y_train, x_test, y_test, epoch):
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
    history = model.fit(x_train, y_train, batch_size=64, epochs=epoch, validation_data=(x_test, y_test),
                        callbacks=[rlrp])
    return history


def print_accuracy(save_dir, model: Sequential, history, x_test, y_test, value):
    # Оценка точности модели на тестовых данных
    accuracy = model.evaluate(x_test, y_test)[1] * 100
    print("Accuracy of our model on test data : ", accuracy, "%")

    epochs = [i for i in range(len(history.history['accuracy']))]
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    # Создание единого графика с двумя подграфиками
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))

    # График потерь
    ax[0].plot(epochs, train_loss, label='Training Loss')
    ax[0].plot(epochs, test_loss, label='Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")

    # График точности
    ax[1].plot(epochs, train_acc, label='Training Accuracy')
    ax[1].plot(epochs, test_acc, label='Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")

    # Сохранение графика
    combined_plot_path = os.path.join(save_dir, f'training_testing_loss_accuracy_{value}_PCA.png')
    plt.tight_layout()  # Уплотнение подграфиков
    plt.savefig(combined_plot_path)
    plt.close(fig)  # Закрытие графика после сохранения

    # Отображение графика
    plt.show()


# %%
# predicting on test data.
def print_confusion_matrix(save_dir, y_pred, y_test, encoder, value):
    y_test = encoder.inverse_transform(y_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)

    df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df['Predicted Labels'] = y_pred.flatten()
    df['Actual Labels'] = y_test.flatten()
    # plt.show()

    # Сохраняем график
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{value}_PCA.png'))
    plt.close()  # Закрываем график, чтобы освободить память

    # result in console
    print(classification_report(y_test, y_pred))


def main():
    file_100k = pd.read_csv('/Users/s.pentin/PycharmProjects/SER_fqw/features/features_100000.csv')
    # print(file_100k.shape)
    # file = pd.read_csv('/Users/s.pentin/PycharmProjects/SER_fqw/features/features_2000.csv')
    layer = 6
    value_pca = "without"  # в процентах компоненты ПСА
    epoch = 100

    X, Y = distribution_data(file_100k)
    encoder, Y = one_hot_encoder(Y)

    x_train, x_test, y_train, y_test = splitting_data(X, Y)

    print(x_train.shape)
    print(x_test.shape)

    x_train, x_test = standard_data(x_train, x_test)

    # x_train_pca, x_test_pca = pca_for_n_component(x_train, x_test, component=value_pca / 100)

    # print(x_train.shape)
    # print(x_test_pca.shape)

    model = create_model(x_train)

    history = fit_model(model, x_train, y_train, x_test, y_test, epoch)

    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)

    # Директория для сохранения графиков
    save_dir = f'/Users/s.pentin/PycharmProjects/SER_fqw/resources/pictures/{layer}_layers_100k'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print_accuracy(save_dir, model, history, x_test, y_test, value_pca)
    print_confusion_matrix(save_dir, y_pred, y_test, encoder, value_pca)


if __name__ == '__main__':
    main()
