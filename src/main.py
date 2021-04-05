"""
Script python pour ouvrir les fichiers de traces de clavier

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from os import listdir
from os.path import isfile, join
import string
import re
import joblib
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from read_pics import get_pics_from_file
from sklearn.model_selection import train_test_split


# Noise reduction
def fft_noise_low_pass(sig, f_sample, lg):
    sig_fft = scipy.fft.rfft(sig)
    power = np.abs(sig_fft) ** 2
    pos_mask = np.where(f_sample > 0)
    freqs = f_sample[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]
    sig_fft[np.abs(f_sample) > peak_freq] = 0
    res = scipy.fft.irfft(sig_fft)
    if len(res) != lg:
        res = np.append(res, res[-1])
    return res


def noise_reduction(trames, info):
    trames = np.array(trames)
    trames_T = trames.transpose()
    freq_sa = scipy.fft.rfftfreq(len(trames), 1 / info["freq_sampling_khz"])
    # plt.plot(freq_sa)
    for i in range(17):
        trames_T[i] = fft_noise_low_pass(trames_T[i], freq_sa, len(trames))
    return trames_T.transpose()


def normalize_trames(trames):
    for i in range(len(trames)):
        trames[i] = trames[i] / np.linalg.norm(trames[i])
    return trames


# Classifier
def new_trained_classifier(save=False, filename_model='classifier.model', noise_mean=None, with_specials=True):
    data, res = [], []
    dico, _ = get_dico()
    for f in listdir('../data'):
        filename = str(join('../data', f))
        letter = re.search('pics_(.+?)\\.bin', f).group(1)
        if isfile(join('../data', f)) and f != 'pics_LOGINMDP.bin' \
                and (
                with_specials or letter in string.digits or letter in string.ascii_uppercase or letter == "NOKEY" or letter == "ENTER"):
            sgl, infos = get_pics_from_file(filename)
            if noise_mean is not None:
                sgl = np.apply_along_axis(sub, 1, sgl, noise_mean)
            sgl = np.array(sgl)
            if not with_specials:
                sgl = np.delete(sgl, [4, 5], 1)
            data += list(sgl)
            res += [dico[letter]] * len(sgl)

    X_train, X_test, y_train, y_test = train_test_split(data, res, test_size=0.3, shuffle=True)

    clf = RandomForestClassifier(n_estimators=30, random_state=1, n_jobs=4)
    clf.fit(X_train, y_train)

    if save:
        joblib.dump(clf, filename_model)

    print(clf.score(X_test, np.array(y_test)))

    return clf


def get_trained_model(filename_model='classifier.model', force_new=False, save_if_new=False):
    res = None
    if force_new:
        return new_trained_classifier(save=save_if_new, filename_model=filename_model)
    try:
        res = joblib.load(filename_model)
    except Exception:
        res = new_trained_classifier(save=save_if_new, filename_model=filename_model)
    return res


def noise_trame():
    trames_noise, _ = get_pics_from_file(f'../data/pics_NOKEY.bin')
    return np.mean(trames_noise, axis=0)


def plot_encodage_lettres():
    plot_encodage(string.ascii_uppercase, noise_trame(), "Lettres")


def plot_encodage_chiffres():
    plot_encodage(string.digits, noise_trame(), "Chiffres")


def plot_encodage_spes():
    plot_encodage(['CTRL', 'ENTER', 'NOKEY', 'SHIFT', 'SPACE', 'SUPPR'], noise_trame(), "Autres")


def plot_encodage(characters_list, noise_trame, title=""):
    res = []
    for c in characters_list:
        trames, infos = get_pics_from_file(f'../data/pics_{c}.bin')
        trames = np.array(trames)
        trames = np.apply_along_axis(sub, 1, trames, noise_trame)
        trames_moyenne_br = np.mean(trames, axis=0)
        res.append(trames_moyenne_br)
    plt.matshow(np.array(res))
    plt.title(title)
    # locs, labels = plt.yticks()
    plt.yticks(range(len(characters_list)), characters_list)
    plt.show()


def sub(line, mean_line):
    return line - mean_line


def get_dico():
    count = 0
    dico = {}
    for c in string.ascii_uppercase:
        dico[c] = count
        count += 1
    for c in string.digits:
        dico[c] = count
        count += 1
    other = ['CTRL', 'ENTER', 'NOKEY', 'SHIFT', 'SPACE', 'SUPPR']
    for c in other:
        dico[c] = count
        count += 1
    inv_dict = {v: k for k, v in dico.items()}
    return dico, inv_dict


if __name__ == "__main__":
    dico_main, inv_dico_main = get_dico()

    # Plot les moyennes des trames en fontion de la lettre qu'elles representent

    plot_encodage_lettres()
    plot_encodage_chiffres()
    plot_encodage_spes()
    # Crée et entraine deux nouveaux classifiers et les sauvegarde
    # clf_without_spe =  new_trained_classifier(save=True, filename_model="clf_without_spe.model", with_specials=False, noise_mean=noise_trame())
    # clf_spe =  new_trained_classifier(save=True, filename_model="clf.model", with_specials=True, noise_mean=noise_trame())

    # Charge les classifiers depuis les fichiers
    clf_without_spe = get_trained_model(filename_model="clf_without_spe.model")
    clf_spe = get_trained_model(filename_model="clf.model")

    trames, infos = get_pics_from_file(f'../data/pics_LOGINMDP.bin')

    trames = np.array(trames)
    trames = np.apply_along_axis(sub, 1, trames, noise_trame())
    trames_without_spe = np.delete(trames, [4, 5], 1)

    res = []
    res_without_spe = []
    for chk in np.split(trames, range(200, len(trames), 200)):
        histo = {k: 0 for k, _ in dico_main.items()}
        predictions = clf_spe.predict(chk)
        for prediction in predictions:
            tmp = inv_dico_main[prediction]
            histo[inv_dico_main[prediction]] += 1
        res.append(sorted(histo.items(), key=lambda item: item[1], reverse=True)[:4])

    for chk in np.split(trames_without_spe, range(200, len(trames), 200)):
        histo = {k: 0 for k, _ in dico_main.items()}
        predictions = clf_without_spe.predict(chk)
        for prediction in predictions:
            tmp = inv_dico_main[prediction]
            histo[inv_dico_main[prediction]] += 1
        res_without_spe.append(sorted(histo.items(), key=lambda item: item[1], reverse=True)[:4])

    for i in range(len(res)):
        print(f"SPE RES: {res[i]} ---- NO SPE RES: {res_without_spe[i]}")
