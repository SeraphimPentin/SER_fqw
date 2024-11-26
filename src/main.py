import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Audio
from IPython.core.display_functions import display
from scipy.signal import lfilter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa
import librosa.display
from sklearn.decomposition import PCA
from keras.src.models import Sequential
from keras.src.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.src.callbacks import ReduceLROnPlateau

# %%
DATA_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\raw_crowd_train.tsv"
AUDIO_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\wavs\\"

DATA_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\raw_crowd_test.tsv"
AUDIO_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\wavs\\"

EPOCH = 5

# %%
data_train = pd.read_csv(DATA_TRAIN, delimiter='\t')
data_test = pd.read_csv(DATA_TEST, delimiter='\t')
data_train.tail()


# data_test.head()
# %%
def create_countplot(data, name):
    plt.title(f'Count of emotions for {name} ', size=16)
    sns.countplot(x='annotator_emo', data=data)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


# %%
create_countplot(data_train, "train")
create_countplot(data_test, "test")


# %%
def create_tsv_file_for_N_records(value: int):
    # Равное количество записей для каждой эмоции
    emotions = ['angry', 'neutral', 'sad', 'positive']
    records_per_emotion = value // len(emotions)

    # Пустой DataFrame для хранения выбранных записей
    selected_data = pd.DataFrame(columns=data_train.columns)

    # Выбор равного количества записей для каждой эмоции
    for emo in emotions:
        emo_data = data_train[(data_train['annotator_emo'] == emo) & (data_train['speaker_emo'] == emo)].head(
            records_per_emotion)
        selected_data = pd.concat([selected_data, emo_data])

    # Путь для сохранения нового файла
    raw_crowd_train = f"C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_{value}.tsv"

    # Сохранение выбранных данных в новом TSV файле
    selected_data.to_csv(raw_crowd_train, sep='\t', index=False)


# %%
# Создаем таблицу на 2000 записей, где каждой аудиозапсиси будет по 500 шт.
create_tsv_file_for_N_records(2000);

# %%
# %%
data_2000_records = pd.read_csv("C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_2000.tsv",
                                delimiter='\t')
data_2000_records.tail()


# %%
def create_waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(emotion), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def create_spectrogram(data, sr, emotion):
    # Преобразование аудиоданных в краткосрочное преобразование Фурье (STFT)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(emotion), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


def create_mfcc_plot(data, sr, emotion):
    # Извлечение признаков MFCC
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)

    # Визуализация графика MFCC
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC for audio with {} emotion'.format(emotion), size=15)
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()


def create_zcr_plot(data, sr, emotion):
    # Извлечение признаков Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=data)

    # Визуализация графика ZCR
    plt.figure(figsize=(8, 4))
    plt.plot(zcr[0], color='b')
    plt.title('Zero Crossing Rate for audio with {} emotion'.format(emotion), size=15)
    plt.xlabel('Frame')
    plt.ylabel('ZCR')
    plt.tight_layout()
    plt.show()


def create_lpc_plot(audio_signal, emotion):
    # Вычисление LPC коэффициентов
    lpc_coefficients = librosa.lpc(audio_signal, order=5)

    # Формирование коэффициентов фильтра
    filter_coefficients = np.hstack([[0], -1 * lpc_coefficients[1:]])

    # Применение фильтрации LPC к сигналу
    filtered_signal = lfilter(filter_coefficients, [1], audio_signal)

    # Построение графика исходного сигнала и предсказанного сигнала
    plt.figure(figsize=(10, 4))
    plt.plot(audio_signal, label='Исходный сигнал')
    plt.plot(filtered_signal, linestyle='--', label='Предсказанный сигнал')
    plt.title('Коэффициенты линейного предсказания для  {} эмоции'.format(emotion))
    plt.xlabel('Отсчет')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.show()


def create_waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(emotion), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def create_spectrogram(data, sr, emotion):
    # Преобразование аудиоданных в краткосрочное преобразование Фурье (STFT)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(emotion), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


def create_audio_display(data):
    display(Audio(data))


def create_mfcc_plot(data, sr, emotion):
    # Извлечение признаков MFCC
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)

    # Визуализация графика MFCC
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC for audio with {} emotion'.format(emotion), size=15)
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()


def create_zcr_plot(data, emotion):
    # Извлечение признаков Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=data)

    # Визуализация графика ZCR
    plt.figure(figsize=(8, 4))
    plt.plot(zcr[0], color='b')
    plt.title('Zero Crossing Rate for audio with {} emotion'.format(emotion), size=15)
    plt.xlabel('Frame')
    plt.ylabel('ZCR')
    plt.tight_layout()
    plt.show()


def create_lpc_plot(data, emotion):
    # Вычисление LPC коэффициентов
    lpc_coefficients = librosa.lpc(data, order=5)

    # Формирование коэффициентов фильтра
    filter_coefficients = np.hstack([[0], -1 * lpc_coefficients[1:]])

    # Применение фильтрации LPC к сигналу
    filtered_signal = lfilter(filter_coefficients, [1], data)

    # Построение графика исходного сигнала и предсказанного сигнала
    plt.figure(figsize=(10, 4))
    plt.plot(data, label='Исходный сигнал')
    plt.plot(filtered_signal, linestyle='--', label='Предсказанный сигнал')
    plt.title('Коэффициенты линейного предсказания для  {} эмоции'.format(emotion))
    plt.xlabel('Отсчет')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.show()


# %%
# Демонстрация фьючерсов, которые в дальнейшем будем извлекать
data_4 = pd.read_csv("C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_4.tsv", delimiter='\t')
for index, row in data_4.iterrows():
    emotion = row['speaker_emo']
    audio_path = AUDIO_TRAIN + row['hash_id'] + ".wav"
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    create_audio_display(audio_path)
    create_waveplot(audio_data, sample_rate, emotion)
    create_spectrogram(audio_data, sample_rate, emotion)
    create_mfcc_plot(audio_data, sample_rate, emotion)
    create_zcr_plot(audio_data, emotion)
    create_lpc_plot(audio_data, emotion)


# %%
# Извлечение признаков
def extract_features(data, sample_rate):
    # signal, sample_rate = librosa.load(path, sr=None)
    result = np.array([])

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    return result


# %%
# Возьмем 1000 первых записей для проверки работы НС
first_1000_rows = data_train.head(1000)
create_countplot(first_1000_rows)

# %%
X, Y = [], []
audio_path = AUDIO_TRAIN + data_2000_records.hash_id + ".wav"
emotions = data_2000_records.annotator_emo

audio_path.count()

# %%
for path, emotion in zip(audio_path, emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        Y.append(emotion)
# %%
Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('C:\\Users\\s5pen\\DataSpellProjects\\SER\\features\\features_2000.csv', index=False)
Features.head()
print("end work")

# %%
# Подготовка данных
X = Features.iloc[:, :-1].values
Y = Features['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

# %%

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%
# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%
# Применение PCA
pca = PCA(n_components=0.90)  # Выбор числа компонент, сохраняющего 90% дисперсии
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

x_train_pca.shape

# %%

model = Sequential()

model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train_pca.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

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

model.add(Dense(units=5, activation='softmax'))  # unites - количество эмоций для обучения
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# %%
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history = model.fit(x_train, y_train, batch_size=64, epochs=EPOCH, validation_data=(x_test_pca, y_test),
                    callbacks=[rlrp])

# %%
print("Accuracy of our model on test data : ", model.evaluate(x_test_pca, y_test)[1] * 100, "%")

epochs = [i for i in range(EPOCH)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20, 6)
ax[0].plot(epochs, train_loss, label='Training Loss')
ax[0].plot(epochs, test_loss, label='Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs, train_acc, label='Training Accuracy')
ax[1].plot(epochs, test_acc, label='Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

# %%
# predicting on test data.
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()
df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

#%%
df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(10)
print(classification_report(y_test, y_pred))
#%%
