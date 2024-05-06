import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
import librosa.display
from scipy.signal import lfilter

# %%
DATA_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\raw_crowd_train.tsv"
AUDIO_TRAIN = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_train\\wavs\\"

DATA_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\raw_crowd_test.tsv"
AUDIO_TEST = "C:\\Users\\s5pen\\YandexDisk\\ВКР\\crowd_test\\wavs\\"


data_train = pd.read_csv(DATA_TRAIN, delimiter='\t')
data_train.tail()

# %%
def create_countplot(data):
    plt.title('Count of emotions for train ', size=16)
    sns.countplot(x='annotator_emo', data=data)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


# %%
create_countplot(data_train)

# %%
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


# %%
create_tsv_file_for_N_records(4)


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

def create_lpc_plot(audio_signal,  emotion):
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
# %%
data_4 = pd.read_csv("C:\\Users\\s5pen\\DataSpellProjects\\SER\\data_tcv\\raw_crowd_train_4.tsv", delimiter='\t')
data_4

# %%  Извлечение признаков

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

        features = np.hstack((zcr, mel_spectrogram, mfcc))#, chroma_stft
        X.append(features)
        Y.append(emotion)

    return np.array(X), np.array(Y)
# %%
audio_path = AUDIO_TRAIN + data_4.hash_id + ".wav"
emotions = data_4.annotator_emo

X, Y = extract_features(audio_path, emotions)