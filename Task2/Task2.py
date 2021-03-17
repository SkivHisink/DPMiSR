import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
#TODO:
# 1) Сделать чтобы точек на графике свёртки было len(a)+len(b)
# 2) Добавить свёртку через сумму
# Аналогично для кросс корелляции
#Частота дискретизации
SAMPLE_RATE = 44100  #44100 Гц
#Длина сгенерированной выборки
DURATION = 5  # Секунды
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi для преобразования в радианы
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

from scipy.fft import ifft
def convolution(a, b):
    return ifft(a * b)
def cross_correlation(a, b):
    return ifft(np.conjugate(a) * b)
#Используем быстрое преобразование Фурье
from scipy.fft import fft, fftfreq
xf = fftfreq(1000, 1 / SAMPLE_RATE)
_, sin_func = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
#Шаг
arg = np.arange(-50, 50, 0.1)
#Прямоугольный сигнал от -10 до 10
def rect(x):
    return 1 if np.abs(x) <= 10 else 0
result = np.zeros(len(arg))
#треугольник в 0
f1 = lambda x: np.maximum(0, 1 - abs(x))
#экспанента с нуля резко затухающая от 1 до нуля по y
f2 = lambda x: (np.exp(-2 * x) * (x > 0)) if abs(x) < 100 else 0
data_f1 = np.zeros(len(arg))
data_f2 = np.zeros(len(arg))
for i in range(0, len(arg)):
    result[i] = rect(arg[i])
    data_f1[i] = f1(arg[i])
    data_f2[i] = f2(arg[i])

#Свёртка
def convolution_test(first_data, second_data, lenth, x):
    fig, axs = plt.subplots(3, 1)
    first_len = len(first_data)
    second_len = len(second_data)
    if first_len != second_len:
        new_len = 0
        if first_len > second_len:
            new_len = first_len
        else:
            new_len = second_len
            tmp = firts_func
            first_data = second_data
            second_data = tmp
        zeroshki = np.zeros(new_len - len(second_data))
        second_data = np.concatenate((second_data, zeroshki))
    point_numb = len(arg)
    a = fft(first_data[:point_numb])
    b = fft(second_data[:point_numb])
    conv = convolution(a, b)
    rev_conv = np.concatenate((conv[int(point_numb / 2):],conv[:int(point_numb / 2)]))
    axs[0].plot(x,first_data[:point_numb])
    axs[1].plot(x,second_data[:point_numb])
    axs[2].plot(x,rev_conv, 'tab:red')
    axs[2].plot(x,np.convolve(first_data[:point_numb], second_data[:point_numb], mode="same"))
    plt.show()
#Кросс корелляция
def сross_corellation_test(first_data, second_data, lenth, x):
    fig, axs = plt.subplots(3, 1)
    first_len = len(first_data)
    second_len = len(second_data)
    if first_len != second_len:
        new_len = 0
        if first_len > second_len:
            new_len = first_len
        else:
            new_len = second_len
            tmp = firts_func
            first_data = second_data
            second_data = tmp
        zeroshki = np.zeros(new_len - len(second_data))
        second_data = np.concatenate((second_data, zeroshki))
    point_numb = lenth
    a = fft(first_data[:point_numb]) 
    b = fft(second_data[:point_numb])
    cross = cross_correlation(a, b)
    rev_cross = np.concatenate((cross[int(point_numb / 2):],cross[:int(point_numb / 2)]))
    axs[0].plot(x, first_data[:point_numb])
    axs[1].plot(x, second_data[:point_numb])
    axs[2].plot(-x, rev_cross, 'tab:red')
    axs[2].plot(x, np.correlate(first_data[:point_numb], second_data[:point_numb], mode="same"))
    plt.show()
argg = np.arange(-50, 50, 0.1)

#Применяем свёртку на синус и прямоугольник
convolution_test(sin_func, result, len(arg), argg)
#Применяем свёртку на прямоугольник и прямоугольник
convolution_test(result, result, len(arg), argg)
#Применяем свёртку на два синуса
convolution_test(sin_func, sin_func, len(arg), argg)
#Применяем свёртку на две функции
convolution_test(data_f1, data_f2, len(arg), argg)
#Применяем Кросс корелляцию на синус и прямоугольник
сross_corellation_test(sin_func, result, len(arg), argg)
#Применяем Кросс корелляцию на прямоугольник и прямоугольник
сross_corellation_test(result, result, len(arg), argg)
#Применяем Кросс корелляцию на два синуса
сross_corellation_test(sin_func, sin_func, len(arg), argg)
#Применяем Кросс корелляцию на две функции
сross_corellation_test(data_f1, data_f2, len(arg), argg)
#Применяем Кросс корелляцию на две функции часть 2
a = [0,0,0,5,5,5,5,5,0,0,0, 0]
xa = np.zeros(len(a))
for i in range(int(-len(a) / 2), int(len(a) / 2)):
    xa[i] = i
v = [0,0,0,5,4,3,2,1]
xa = np.sort(xa)
сross_corellation_test(a, v, len(a), xa)
#Применяем Свёртку на две функции
convolution_test(a, v, len(a), xa)
#noise
mixed_tone = np.zeros(len(arg))
for i in range(0, len(arg)):
    mixed_tone[i] = np.random.uniform(10) #normal -Gaussian
#Вычетание среденего для устранения пика у преобразования Фурье
mixed_tone-=np.mean(mixed_tone)
#Применяем Кросс корелляцию на шум и синус
сross_corellation_test(mixed_tone, result, len(arg), argg)
#Применяем Свёртку на шум и синус
convolution_test(mixed_tone, result, len(arg), argg)

#Чужая функция тест
t = np.arange(-5, 5, 0.1)
convolution_test(np.cos(t/4*1.2+np.pi/2), np.cos(t/4*1.2+np.pi/2), len(t), t)
сross_corellation_test(np.cos(t/4*1.2+np.pi/2), np.cos(t/4*1.2+np.pi/2), len(t), t)