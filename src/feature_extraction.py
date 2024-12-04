import librosa
from librosa import feature
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os.path

# Windows
# %%
# DATA_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\raw_crowd_train.tsv"
# DATA_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\raw_crowd_train.tsv"

# MacOS
DATA_TRAIN = "/Users/s.pentin/Yandex.Disk.localized/Учеба/ДИПЛОМ/DataSet/crowd/crowd_train/raw_crowd_train.tsv"
AUDIO_TRAIN = "/Users/s.pentin/Yandex.Disk.localized/Учеба/ДИПЛОМ/DataSet/crowd/crowd_train/wavs/"

# DATA_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\raw_crowd_test.tsv"
# AUDIO_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\wavs\\"

EPOCH = 5

# %%
data_train = pd.read_csv(DATA_TRAIN, delimiter='\t')


# print(data_train.tail(10))
# print(data_train.shape)

# это можно использовать, когда скачаны все aудио файлы на ПК
# def create_tsv_file_for_N_records(value: int):
#     # Равное количество записей для каждой эмоции
#     emotions = ['angry', 'neutral', 'sad', 'positive']
#     records_per_emotion = value // len(emotions)
#
#     # Пустой DataFrame для хранения выбранных записей
#     selected_data = pd.DataFrame(columns=data_train.columns)
#
#     # Выбор равного количества записей для каждой эмоции
#     for emo in emotions:
#         emo_data = data_train[(data_train['annotator_emo'] == emo) & (data_train['speaker_emo'] == emo)].head(
#             records_per_emotion)
#
#         # Генерация путей для аудиофайлов
#         audio_paths = AUDIO_TRAIN + emo_data.hash_id + ".wav"
#         print(audio_paths)
#
#         # Проверка существования файлов
#         for audio_path in audio_paths:
#             if os.path.isfile(audio_path):
#                 selected_data = pd.concat([selected_data, emo_data])
#
#     # Путь для сохранения нового файла
#     raw_crowd_train = f"/Users/s.pentin/PycharmProjects/SER_fqw/data_tcv/raw_crowd_train_{value}.tsv"
#
#     # Сохранение выбранных данных в новом TSV файле
#     selected_data.to_csv(raw_crowd_train, sep='\t', index=False)


# делает доп проверку, что файл существует (скачан)
def create_tsv_file_for_N_records(value: int):
    # Равное количество записей для каждой эмоции
    emotions = ['angry', 'neutral', 'sad', 'positive']
    records_per_emotion = value // len(emotions)

    # Пустой DataFrame для хранения выбранных записей
    selected_data = pd.DataFrame(columns=data_train.columns)

    # Выбор равного количества записей для каждой эмоции
    for emo in emotions:
        emo_data = data_train[(data_train['annotator_emo'] == emo) & (data_train['speaker_emo'] == emo)]

        # Переменная для хранения найденных записей
        found_records = 0

        for index, row in emo_data.iterrows():
            # Генерируем список возможных имен файлов на основе hash_id
            possible_hash_ids = [row['hash_id'], row['hash_id'] + '1', row['hash_id'] + '2']  # Пример вариаций
            for hash_id in possible_hash_ids:
                audio_path = AUDIO_TRAIN + hash_id + ".wav"
                print(f"Checking if file exists: {audio_path}")  # Проверка существования файла

                if os.path.isfile(audio_path):
                    selected_data = pd.concat([selected_data, row.to_frame().T])  # Добавляем строку
                    found_records += 1
                    break  # Выходим из цикла, если файл найден

            # Если уже нашли нужное количество записей, выходим из цикла
            if found_records >= records_per_emotion:
                break

    # Путь для сохранения нового файла
    raw_crowd_train = f"/Users/s.pentin/PycharmProjects/SER_fqw/data_tcv/raw_crowd_train_{value}.tsv"

    # Сохранение выбранных данных в новом TSV файле
    selected_data.to_csv(raw_crowd_train, sep='\t', index=False)


def start():
    create_tsv_file_for_N_records(100_000)
    # create_tsv_file_for_N_records(10)


# data_10_records = pd.read_csv("/Users/s.pentin/PycharmProjects/SER_fqw/data_tcv/raw_crowd_train_10.tsv", delimiter='\t')
#
# print(data_10_records.shape)
# print(data_10_records.tail(10))


data_100000_records = pd.read_csv("/Users/s.pentin/PycharmProjects/SER_fqw/data_tcv/raw_crowd_train_100000.tsv",
                                  delimiter='\t')

print(data_100000_records.tail(2))
print(data_100000_records.shape)


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    return result


def create_countplot(data, name):
    plt.title(f'Count of emotions for {name} ', size=16)
    sns.countplot(x='annotator_emo', data=data)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


# Извлечение признаков
def extract_zcr(data):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    return zcr


def extract_mel_spectrogram(data, sample_rate):
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    return mel_spectrogram


def extract_mfcc(data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    return mfcc


def extract_chroma_stft(data, sample_rate):
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    return chroma_stft


def extract_features(data_path, emotions):
    X, Y = [], []
    for path, emotion in zip(data_path, emotions):
        data, sample_rate = librosa.load(path)

        zcr = extract_zcr(data)
        mel_spectrogram = extract_mel_spectrogram(data, sample_rate)
        mfcc = extract_mfcc(data, sample_rate)

        # chroma_stft = extract_chroma_stft(data, sample_rate)

        features = np.hstack((zcr, mel_spectrogram, mfcc))  # , chroma_stft
        X.append(features)
        Y.append(emotion)

    return np.array(X), np.array(Y)


# data_10_records = pd.read_csv("/Users/s.pentin/PycharmProjects/SER_fqw/data_tcv/raw_crowd_train_10.tsv", delimiter='\t')

audio_path = AUDIO_TRAIN + data_100000_records.hash_id + ".wav"
emotions = data_100000_records.annotator_emo

X, Y = extract_features(audio_path, emotions)

Features = pd.DataFrame(X)
Features['labels'] = Y
# Сохранение в CSV
Features.to_csv('/Users/s.pentin/PycharmProjects/SER_fqw/features/features_100000.csv', index=False)

print(audio_path.count())
print(Features.shape)
