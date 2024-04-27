#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Audio
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import librosa.display
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.src.callbacks import ReduceLROnPlateau

#%%
DATA_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\raw_crowd_train.tsv"
AUDIO_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\wavs\\"

DATA_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\raw_crowd_test.tsv"
AUDIO_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\wavs\\"

EPOCH = 3

data_train = pd.read_csv(DATA_TRAIN, delimiter='\t')
data_train.tail()
#%%
def create_countplot(data):
    plt.title('Count of emotions  for train ', size=16)
    sns.countplot(x='annotator_emo', data=data)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()
#%%
create_countplot(data_train)
#%%
data_test = pd.read_csv(DATA_TEST, delimiter='\t')
data_test.head()
#%%
create_countplot(data_test)
#%%
def create_tsv_file_for_N_records(value: int):
    # Равное количество записей для каждой эмоции
    emotions = ['angry', 'neutral', 'sad', 'positive']
    records_per_emotion = value // len(emotions)

    # Пустой DataFrame для хранения выбранных записей
    selected_data = pd.DataFrame(columns=data_train.columns)

    # Выбор равного количества записей для каждой эмоции
    for emo in emotions:
        emo_data = data_train[(data_train['annotator_emo'] == emo) & (data_train['speaker_emo'] == emo)].head(records_per_emotion)
        selected_data = pd.concat([selected_data, emo_data])

    # Путь для сохранения нового файла
    raw_crowd_train = f"C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_{value}.tsv"

    # Сохранение выбранных данных в новом TSV файле
    selected_data.to_csv(raw_crowd_train, sep='\t', index=False)

#%%
create_tsv_file_for_N_records(5000)
#%%
def create_waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(emotion), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def create_spectrogram(data, sr, emotion):
    # Преобразование aудиоданных в краткосрочное преобразование Фурье (STFT)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(emotion), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

def create_mel_spectrogram(data, sr, emotion):
    # Вычисление мел-спектрограммы
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)

    # Построение графика мел-спектрограммы
    plt.figure(figsize=(12, 3))
    plt.title('Mel-Spectrogram for audio with {} emotion'.format(emotion), size=15)
    librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f дБ')
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


def create_lfpc_plot(data, sr, emotion):
    lfpc = librosa.feature.melspectrogram(y=data, sr=sr, power=2.0)
    lfpc_db = librosa.power_to_db(lfpc, ref=np.max)

    plt.figure(figsize=(12, 3))
    plt.title('LFPC for audio with {} emotion'.format(emotion), size=15)
    plt.imshow(lfpc_db, origin='lower', aspect='auto', cmap='inferno', interpolation='nearest')
    plt.colorbar(format='%+2.0f дБ')
    plt.xlabel('Time')
    plt.ylabel('Mel bins')
    plt.tight_layout()
    plt.show()

#%%
data_4 = pd.read_csv("C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_4.tsv", delimiter='\t')
data_4
#%%
for index, row in data_4.iterrows():
    emotion = row['speaker_emo']
    audio_path = AUDIO_TRAIN + row['hash_id'] + ".wav"
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    # create_audio_display(audio_path)
    # create_waveplot(audio_data, sample_rate, emotion)
    # create_spectrogram(audio_data, sample_rate, emotion)
    create_mel_spectrogram(audio_data, sample_rate, emotion)
    create_mfcc_plot(audio_data, sample_rate, emotion)
    # create_zcr_plot(audio_data, sample_rate, emotion)
    create_lfpc_plot(audio_data, sample_rate, emotion)

#%% md
# Извлечение признаков
#%%
def extract_features(data, sample_rate):
    # signal, sample_rate = librosa.load(path, sr=None)
    result = np.array([])

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    # # LFPC - log frequency power coefficients
    # lfpc = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, power=2.0).T, axis=0)
    # lfpc_db = librosa.power_to_db(lfpc, ref=np.max)
    # result = np.hstack((result, lfpc_db)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally


    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, offset=0.6)

    result = np.array(extract_features(data, sample_rate))

    return result

#%%
first_1000_rows = data_train.head(1000)
create_countplot(first_1000_rows)
#%%
data_1000_records = pd.read_csv("C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_1000.tsv", delimiter='\t')
data_2000_records = pd.read_csv("C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_2000.tsv", delimiter='\t')
data_5000_records = pd.read_csv("C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_5000.tsv",delimiter='\t')
data_10_000_records = pd.read_csv("C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_10000.tsv",delimiter='\t')

#%%
create_countplot(data_1000_records)
#%%
X, Y = [], []
audio_path = AUDIO_TRAIN + data_1000_records.hash_id + ".wav"
emotions = data_1000_records.annotator_emo
for path, emotion in zip(audio_path , emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        Y.append(emotion)

# def extract_features_and_labels(audio_path, emotions):
#     X, Y = [], []
#     for path, emotion in zip(audio_path, emotions):
#         feature = get_features(path)
#         for ele in feature:
#             X.append(ele)
#             Y.append(emotion)
#     return X, Y
#%%
# X, Y = [], []
# audio_path = AUDIO_TRAIN + data_1000_records.hash_id + ".wav"
# emotions = data_1000_records.annotator_emo
#
# for path, emotion in zip(audio_path, emotions):
#     features = get_features(path)  # Извлечение признаков для текущего аудиофайла
#     X.append(features)  # Добавление массива признаков в X
#     Y.append(emotion)   # Добавление метки эмоции в Y
#
# # Преобразование в массив numpy для удобства работы
# X = np.array(X)
# Y = np.array(Y)
#
# # Проверим размерности X и Y
# print("Shape of X:", X.shape)
# print("Shape of Y:", Y.shape)
#%%
# from sklearn.decomposition import PCA
# pca = PCA(n_components=0.90)  # Выбор числа компонент, сохраняющего 90% дисперсии
# X_pca = pca.fit_transform(X)
#
# # Проверим новую размерность X после применения PCA
# print("Shape of X after PCA:", X_pca.shape)
#%%
# #Грфик накопленной дисперсии -
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
#
# # Построение графика
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_variance, marker='o', linestyle='-')
# plt.title('Накопленная дисперсия, объясненная главными компонентами')
# plt.xlabel('Количество главных компонент')
# plt.ylabel('Накопленная дисперсия, объясненная')
# plt.grid(True)
# plt.show()
#%%
len(X), len(Y), audio_path.shape
#%%
Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv(f'C:\\Users\\s5pen\\DataSpellProjects\\SER\\features\\features_1000.csv', index=False)
Features.head()

#%%
# #Второй набор
#
# first_10_000_rows = data_train.head(10000)
# # create_countplot(first_10_000_rows)
#
# X, Y = [], []
# audio_path = AUDIO_TRAIN + first_10_000_rows.hash_id + ".wav"
# emotions = first_10_000_rows.annotator_emo
#
# for path, emotion in zip(audio_path , emotions):
#     feature = get_features(path)
#     for ele in feature:
#         X.append(ele)
#         Y.append(emotion)
#
# Features = pd.DataFrame(X)
# Features['labels'] = Y
# Features.to_csv('C:\\Users\\s5pen\\DataSpellProjects\\SER\\features\\features_10000.csv', index=False)
# Features.head()
#%% md
### Подготовка данных
#%%
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
#%%
# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
#%%
from sklearn.model_selection import train_test_split

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_train
#%% md

#%%
# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_train

#%%
# from sklearn.decomposition import PCA
#
# # Применение PCA
# pca = PCA(n_components=0.9)  # Выбор числа компонент, сохраняющего 90% дисперсии
# x_train_pca = pca.fit_transform(x_train)
# x_test_pca = pca.transform(x_test)
#
# x_train_pca.shape, x_train_pca
#%%
model=Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=4, activation='softmax')) # unites - количество эмоций для обучения
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
#%%
x_test.shape, y_test.shape
#%%
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history=model.fit(x_train, y_train, batch_size=64, epochs=EPOCH, validation_data=(x_test, y_test), callbacks=[rlrp])
#%%
print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(EPOCH)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()
#%%
# predicting on test data.
pred_test = model.predict(x_test_pca)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)
#%%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()
#%%
df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(10)
#%%
df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(10)
#%%
print(classification_report(y_test, y_pred))