import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy import signal
from scipy.signal import find_peaks
import copy

# about not irradiate
Google_Drive_datadir = "/path/to/data/experiment/spectra/kobayashi/NaCl/"
DATE = "2022-03-18"
FILE_ID_IRRADIATED_UP = "16"
DATA_FILE_PATH_IRRADIATED_UP = f"{Google_Drive_datadir}/{DATE}/{DATE}_{FILE_ID_IRRADIATED_UP}.txt"
print(DATA_FILE_PATH_IRRADIATED_UP)
FILE_ID_IRRADIATED_BOTTOM = "18"
DATA_FILE_PATH_IRRADIATED_BOTTOM = f"{Google_Drive_datadir}/{DATE}/{DATE}_{FILE_ID_IRRADIATED_BOTTOM}.txt"
print(DATA_FILE_PATH_IRRADIATED_BOTTOM)
FILE_ID_BEFORE_IRRADIATED_UP = "13"
DATA_FILE_PATH_BEFORE_IRRADIATED_UP = f"{Google_Drive_datadir}/{DATE}/{DATE}_{FILE_ID_BEFORE_IRRADIATED_UP}.txt"
print(DATA_FILE_PATH_BEFORE_IRRADIATED_UP)
FILE_ID_BEFORE_IRRADIATED_BOTTOM = "15"
DATA_FILE_PATH_BEFORE_IRRADIATED_BOTTOM = f"{Google_Drive_datadir}/{DATE}/{DATE}_{FILE_ID_BEFORE_IRRADIATED_BOTTOM}.txt"
print(DATA_FILE_PATH_BEFORE_IRRADIATED_BOTTOM)
# File_Location = "/Users/kobayashiaiketsu/Library/CloudStorage/GoogleDrive-a.kobayashi267@gmail.com/.shortcut-targets-by-id/1ZkGf0KEeIpXHgpjgZ6RIBuECFYX3dERR/study/data/experiment/spectra/kobayashi/90min&5min_Absorbtion_depth/"
# outfile_fig = File_Location + 'Absorption_depth_90min.png'
IRRADIATION_TIME = "5min"


def read_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t', skiprows=14)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def normalize_data(y, target_index):
    norm_y = y / y[target_index]
    return norm_y


# Specify the target values for normalization # Specify the desired x1 value
# x =1000.248で規格化

target_index = 1000


wavelengths_irradiated, intensities_irradiated_up = read_data(
    DATA_FILE_PATH_IRRADIATED_UP)
wavelengths_irradiated, intensities_irradiated_bottom = read_data(
    DATA_FILE_PATH_IRRADIATED_BOTTOM)
wavelengths_before_irradiated, intensities_before_irradiated_up = read_data(
    DATA_FILE_PATH_BEFORE_IRRADIATED_UP)
wavelengths_before_irradiated, intensities_before_irradiated_bottom = read_data(
    DATA_FILE_PATH_BEFORE_IRRADIATED_BOTTOM)


# Normalize intensities
norm_intensities_irradiated_up = normalize_data(
    intensities_irradiated_up, target_index)
norm_intensities_irradiated_bottom = normalize_data(
    intensities_irradiated_bottom, target_index)
norm_intensities_before_irradiated_up = normalize_data(
    intensities_before_irradiated_up, target_index)
norm_intensities_before_irradiated_bottom = normalize_data(
    intensities_before_irradiated_bottom, target_index)


wavelengths = wavelengths_irradiated
intensities = (norm_intensities_irradiated_up + norm_intensities_irradiated_bottom) / \
    (norm_intensities_before_irradiated_up +
     norm_intensities_before_irradiated_bottom)

# プロットするwavelengthの範囲を設定
wavelength_range = (230, 1100)
mask = (wavelengths >= wavelength_range[0]) & (
    wavelengths <= wavelength_range[1])
wavelengths_cut = wavelengths[mask]
intensities_cut = intensities[mask]

# ---多項式関数を表すpolynomial_funcの定義----


def polynomial_func(X, *params):
    if len(params) == 1:
        params = params[0]
    Y = np.zeros_like(X)
    for n, param in enumerate(params):
        Y += param*X**n
    return Y


def trim_wavelengths_without_absorption(wavelengths_cut, intensities_cut, trim_wavelength_ranges):
    x1_min, x1_max, x2_min, x2_max = trim_wavelength_ranges
    x1, y1 = zip(*[(i, j) for i, j in zip(wavelengths_cut,
                 intensities_cut) if (x1_min <= i <= x1_max)])
    x2, y2 = zip(*[(i, j) for i, j in zip(wavelengths_cut,
                 intensities_cut) if (x2_min <= i <= x2_max)])
    trimmed_wavelengths = np.hstack([x1, x2])
    trimmed_intensities = np.hstack([y1, y2])
    return trimmed_wavelengths, trimmed_intensities


# 適宜調整
x1_min = 300
x1_max = 400
x2_min = 900
x2_max = 950

trim_wavelength_ranges = (x1_min, x1_max, x2_min, x2_max)
trimmed_wavelengths, trimmed_intensities = trim_wavelengths_without_absorption(
    wavelengths_cut, intensities_cut, trim_wavelength_ranges)


def polynomial_fitting_trimmed_spectra(wavelengths_cut, intensities_cut, trimmed_wavelengths, trimmed_intensities, dimensions):
    polynomial_fitted_values_list = []
    polynomial_params_list = []
    intensities_divided_by_fitted_values_reversed = []
    for dimension in dimensions:
        polynomial_params = []
        p0 = [1] * (dimension + 1)
        param, cov = curve_fit(
            polynomial_func, trimmed_wavelengths, trimmed_intensities, p0)
        polynomial_params.append(param)
        polynomial_params_list.append(polynomial_params)
        polynomial_fitted_values_list.append(
            polynomial_func(wavelengths_cut, *polynomial_params[0]))
        intensities_divided_by_fitted_value_reversed = 1 - \
            intensities_cut/polynomial_fitted_values_list[dimension-2]
        intensities_divided_by_fitted_values_reversed.append(
            intensities_divided_by_fitted_value_reversed)
    return polynomial_fitted_values_list, polynomial_params_list, intensities_divided_by_fitted_values_reversed


# 2次、3次、4次の多項式でフィッティング
dimensions_to_fit = [1, 2, 3, 4, 5]
polynomial_fitted_values_list, polynomial_params_list, intensities_divided_by_fitted_values_reversed = polynomial_fitting_trimmed_spectra(
    wavelengths_cut, intensities_cut, trimmed_wavelengths, trimmed_intensities, dimensions_to_fit)
opt_dimension = 5  # 適切な次数に変更


def Gaussian_list(x, *params):
    global peak_positions_list
    num_func = int(len(params) / 3)
    y_list = np.zeros_like(x)
    for i in range(num_func):
        param_range = list(range(3 * i, 3 * (i + 1), 1))
        amp = params[int(param_range[0])]
        # ctr = params[int(param_range[1])]
        ctr = peak_positions_list[i]
        wid = params[int(param_range[2])]
        y_list += amp * np.exp(-((x - ctr) / wid) ** 2)
    y_list += params[-1]  # バックグラウンド
    return y_list


def gaussian_fitting(wavelengths_cut, peak_positions_list, peak_list, opt_dimension, intensities_divided_by_fitted_values):
    num_func = len(peak_positions_list) * 3  # ガウシアン関数の数
    gaussian_params = []
    for peak_position, peak in zip(peak_positions_list, peak_list):
        gaussian_params.extend([peak, peak_position, 10.0])  # デフォルトの初期値
    gaussian_params.append(0)  # BKG バックグラウンド
   # ガウシアンフィッティング時に初期値として自動補正後のピーク位置を使用
    params_gauss, cov_gauss = curve_fit(
        Gaussian_list, wavelengths_cut, intensities_divided_by_fitted_values[opt_dimension - 2], gaussian_params, maxfev=50000)

    gaussian_fitted_values = Gaussian_list(wavelengths_cut, *params_gauss)
    return gaussian_fitted_values, params_gauss


# ガウシアンフィッティングの結果から面積を算出する関数
def calculate_gaussian_area(amp, ctr, wid):
    # ガウス積分の公式を用いて面積を計算
    area = amp * wid * np.sqrt(np.pi / 2)
    return area


peak_positions_list = [460, 590, 735, 826]  # ピーク位置を適宜追加
peak_list = [0., 0., 0., 0.]
gaussian_fitted_values, params_gauss = gaussian_fitting(
    wavelengths_cut, peak_positions_list, peak_list, opt_dimension, intensities_divided_by_fitted_values_reversed)

# ガウシアンフィッティング結果からピーク位置を取得
peaks, _ = find_peaks(gaussian_fitted_values)
# 自動補正したピーク位置を表示
auto_corrected_peak_positions = wavelengths_cut[peaks]
for i, peak_position in enumerate(auto_corrected_peak_positions):
    print(f"自動補正後のガウシアン{i+1}のピーク位置: {peak_position}")
# ガウシアンフィッティングの実行
gaussian_fitted_values, params_gauss = gaussian_fitting(
    wavelengths_cut, auto_corrected_peak_positions, peak_list, opt_dimension, intensities_divided_by_fitted_values_reversed)


# グラフのタイトルと軸ラベル
fig = plt.figure(tight_layout=True, figsize=[8, 8])
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.set_title('Experiment Data')
ax2.set_title('Polynomial Fitting')
ax3.set_title('Gaussian Fitting')
title = IRRADIATION_TIME
xl = 'Wavelength [nm]'
yl = 'Reflectance relative to reference'
fig.suptitle(title, fontsize=20)
fig.supxlabel(xl, fontsize=20)
fig.supylabel(yl, fontsize=20)


# グラフのプロット
ax1.plot(wavelengths, intensities)
ax2.plot(wavelengths_cut, intensities_cut)
for i, dimension in enumerate(dimensions_to_fit):
    ax2.plot(
        wavelengths_cut, polynomial_fitted_values_list[i], label=f'Poly {dimension} Fitted')
# ガウシアンフィッティングの実行
gaussian_fitted_values, params_gauss = gaussian_fitting(
    wavelengths_cut, peak_positions_list, peak_list, opt_dimension, intensities_divided_by_fitted_values_reversed)
# グラフのプロット (ax3でのみwavelengthの範囲を指定)
ax3.plot(
    wavelengths_cut, intensities_divided_by_fitted_values_reversed[opt_dimension - 2])
ax3.plot(wavelengths_cut, gaussian_fitted_values, label='Gaussian fit')
# ピーク位置を自動補正して表示
for i, peak_position in enumerate(auto_corrected_peak_positions):
    ax3.axvline(peak_position, linestyle='--', color='red',
                label=f'Peak {i+1} ({peak_position} nm)')

num_func = int(len(params_gauss) / 3)
for i in range(num_func):
    y_list = np.zeros_like(wavelengths_cut)
    param_range = list(range(3 * i, 3 * (i + 1), 1))
    amp = params_gauss[int(param_range[0])]
    ctr = params_gauss[int(param_range[1])]
    wid = params_gauss[int(param_range[2])]
    y_list += amp * np.exp(-((wavelengths_cut - ctr) / wid) ** 2)
    y_list += params_gauss[-1]  # バックグラウンド
    gaussian_fitted_single = amp * \
        np.exp(-((wavelengths_cut - ctr) / wid) ** 2)
    area_single = calculate_gaussian_area(amp, ctr, wid)
    print(f"ガウシアン{i+1}の面積: {area_single}")
    # 面積に基づいて薄く色付け
    color = cm.viridis(i / num_func)  # 適切なカラーマップを選択
    ax3.fill_between(wavelengths_cut, 0, y_list, color=color, alpha=0.3,
                     label=f'Gaussian {i+1} (Area: {area_single:.2f})')


# 軸設定
ax1.set_xlim(200, 1100)
ax1.set_ylim(0.5, 1.5)
ax1.minorticks_on()
ax1.hlines(y=1., xmin=200, xmax=1100, color='black')
ax2.set_xlim(300, 950)
ax2.set_ylim(0.5, 1.5)
ax2.minorticks_on()
ax2.vlines(x=x1_min, ymin=0, ymax=5, color='black')
ax2.vlines(x=x1_max, ymin=0, ymax=5, color='black')
ax2.vlines(x=x2_min, ymin=0, ymax=5, color='black')
ax2.vlines(x=x2_max, ymin=0, ymax=5, color='black')
ax3.set_xlim(300, 950)
ax3.set_ylim(0., 0.2)
ax3.minorticks_on()


# 凡例の表示
ax1.legend()
ax2.legend()
ax3.legend()

# グラフの保存
# plt.savefig(outfile_fig)

# グラフの表示
plt.show()
plt.close()
