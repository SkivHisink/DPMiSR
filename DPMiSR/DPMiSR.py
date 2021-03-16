import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
#TODO: 
#1) Самому вычислять значения по x(freq) - Сделано, за это отвечает массив myxf в функции signaltest
#2) Вычесть среднее из шума, чтобы изабвиться от пика в нуле - Сделано, нужно расскоментить строку mixed_tone-=np.mean(mixed_tone)
#3) Поиграть с размерами ступеньки и посмотреть что будет происходить с преобразованием Фурье - если тоньше или ниже ступенька, то пик уменьшается. 
#В первом случае последующие пики более выразительные. Во втором случае вторичные пики схожи с начальным графиком. 

#Discrete Fourier Translate implementation
def DFT(data, n):
    return sum([data[k] * np.exp(-2j * np.pi * k * n / len(data)) for k in range(0, len(data))])
#Inverse Discrete Fourier Translate implementation
def IDFT(data, k):
    return 1 / len(data) * sum([data[n] * np.exp(2j * np.pi * k * n / len(data)) for n in range(0, len(data))])

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

#Используем быстрое преобразование Фурье для удаления шума
from scipy.fft import fft, fftfreq

# число точек в normalized_tone
N = SAMPLE_RATE * DURATION
from scipy.fft import ifft

def signaltest(normalized_tone):
    myx = [DFT(normalized_tone[:1000], n) for n in range(0, len(normalized_tone[:1000]))]
    yf = fft(normalized_tone[:1000])
    new_sig = ifft(yf)
    xf = fftfreq(1000, 1 / SAMPLE_RATE)
    #Freq Kotelnikova
    myfreq=SAMPLE_RATE/2
    myxf=np.arange(0,myfreq, 2*myfreq/1000)
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Task 1")
   # axs[0][0].set_title("Fast Fourirer Transform")
    axs[0][0].plot(xf[:500], np.abs(yf)[:500])
    #axs[1][0].set_title("Discrete(my) Fourirer Transform")
    axs[1][0].plot(myxf, np.abs(myx)[:500], "tab:green")
    #axs[2][0].set_title("Discrete(my) and Fast FT on one graphic")
    axs[2][0].plot(xf[:500], np.abs(yf)[:500])
    axs[2][0].plot(myxf, np.abs(myx)[:500], "tab:green")
    kyx = [IDFT(myx, k) for k in range(0, len(myx))]
   # axs[0][1].set_title("Inverse Fast Fourirer Transform")
    axs[0][1].plot(new_sig[:1000]) 
   # axs[1][1].set_title("Inverse Discrete(my) Fourirer Transform")
    axs[1][1].plot(kyx, "tab:green")
  #  axs[2][1].set_title("Inverse Fast and Discrete(my) FT")
    #axs[2][1].plot(new_sig[:1000])
    axs[2][1].plot(kyx, "tab:green")
  #  axs[3][0].set_title("Full function")
    axs[3][0].plot(normalized_tone)
  #  axs[3][1].set_title("Source signal for transform")
    axs[3][1].plot(normalized_tone[:1000])
    plt.show()

# Генерируем волну с частотой 2 Гц, которая длится 5 секунд
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

#Смешивание аудиосигналов
_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
mixed_tone = noise_tone 

#clear signal
#умножение на 32767 масштабирует сигнал между -32767 и 32767, 
#что примерно соответствует диапазону np.int16. 
#Код отображает только первые 1000 точек, чтобы мы могли четче проследить структуру сигнала.
normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)
#Используем сигнал 4кГц, на графике преобразования Фурье как раз виден пик в 4000 
signaltest(normalized_tone)
#noise
for i in range(0, len(mixed_tone)):
    mixed_tone[i] = np.random.uniform(10) #normal -Gaussian
#Вычетание среденего для устранения пика
#mixed_tone-=np.mean(mixed_tone)
normalized_tone = np.int16(((mixed_tone) / mixed_tone.max()) * 32767)
#signaltest(normalized_tone)
#summ of signals
#синусоидальный сгенерированный тон 400 Гц, искаженный тоном 4000 Гц
mixed_tone = noise_tone * 0.1 + nice_tone
normalized_tone = np.int16(((mixed_tone) / mixed_tone.max()) * 32767)
#signaltest(normalized_tone)
#rectangle signal
arg = np.arange(-50, 50, 0.1)
result = np.zeros(len(arg))
for i in range(0, len(arg)):
    result[i] = (arg[i] >= - np.abs(arg[0]) / 4 and arg[i] <= np.abs(arg[0]) / 4)
    
signaltest(result)