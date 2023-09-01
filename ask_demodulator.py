#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu August 31 08:06:00 2023

@author: kkn
"""

from os.path import dirname, join as pjoin
from scipy.io import wavfile
import pdb
import scipy.io
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os

# Input wav file
# Имя входного wav файла
wav_file_name = 'wav\\ask_out.wav'
wav_file_name = os.path.join(os.path.dirname(__file__), wav_file_name)

# Name of the output data file
# Имя выходного файла с данными
data_file_out_name = 'received_data.txt'
data_file_out_name = os.path.join(os.path.dirname(__file__), data_file_out_name)

# Sampling rate of wav file, other values don't work
# Частота дискретизации файла wav
wav_samplerate = 44100

# Carrier frequency value for transmission
# Значение частоты для передачи
carrier_frequency = 1764

# Data transfer rate, bits per second
# Скорость передачи данных
transmit_bitrate = 147

# Amplitudes for signal value 0 and 1, in the range from 0 to 1
# Амплитуды для значения сигнала 0 и 1, в диапазоне от 0 до 1
bit_0_amplitude = 0
bit_1_amplitude = 1

# Carrier frequency duration in bit durations, transmitted before the preamble
# Used for PLL operation and signal detection
# Размер несущей в битах
carrier_bit_size = 24

# Value for carrier amplitude transmitted before preamble, bit value = 0 or 1
# Значение для амплитуды несущей передаваемой до преамбулы
carrier_bit_value = 1

# Preamble (alternating frequency) duration, in bit durations
# Used for bit synchronization
# Размер преамбулы, бит
preamble_bit_size = 32

# Postamble duration, only needed for Windows player, loses end of wav file
# Размер постамбулы, бит
postamble_bit_size = 24

# Synchronization word size, bits. Used for byte synchronization
# Размер слова синхронизации, бит
synchronization_word_bit_size = 8

# Sync word bits value
# Значение бит слова синхронизации
synchronization_word_bit_value = 1

# Число отсчётов частоты дискретизации на каждый бит
# Число отсчётов частоты дискретщизации на каждый бит
bit_samplerate_period_per_bit = wav_samplerate / transmit_bitrate

# Open wav file for reading
# Читаем входной файл
input_signal_samplerate, input_signal_data = wavfile.read(wav_file_name)
input_signal_length = input_signal_data.shape[0]

# Signal duration, bits and bytes
# Длительность посылки бит и байт
input_signal_bit_length = input_signal_length / bit_samplerate_period_per_bit
input_signal_byte_length = input_signal_bit_length / 8

# Output parameters to the console
# Вывод параметров
print("Частота дискретизации wav =", wav_samplerate)
print("Частота несущей =", carrier_frequency)
print("Число периодов дискрертизации на бит =", bit_samplerate_period_per_bit);
print("Число отсчётов входного файла =", input_signal_length)
print("Длительность входных данных секунд =", input_signal_length/input_signal_samplerate)
print("Размер несущей, бит =", carrier_bit_size)
print("Значение несущей несущей, бит =", carrier_bit_value)
print("Размер преамбулы, бит =", preamble_bit_size)
print("Размер слова синхронизации, бит =", synchronization_word_bit_size)
print("Значение бит слова синхронизации =", synchronization_word_bit_value)
print("Длина посылки в битах =", input_signal_bit_length)
print("Длина посылки в байтах =", input_signal_byte_length)

# Calculate an array with sinusoid samples
# Формируем отсчёты синусоид и косинусоид для частоты несущей
# Формируем отсчёты синусоид и косинусоид двух частот для одного бита
carrier_sin_samples_index = np.arange(bit_samplerate_period_per_bit)
carrier_sin_samples = np.sin(2 * np.pi * (carrier_frequency / transmit_bitrate) * carrier_sin_samples_index / bit_samplerate_period_per_bit)
carrier_cos_samples = np.cos(2 * np.pi * (carrier_frequency / transmit_bitrate) * carrier_sin_samples_index / bit_samplerate_period_per_bit)

# Threshold value to determine bit value
# Значение порога для определения значения бита
bit_level_threshold = (bit_1_amplitude - bit_0_amplitude) / 2

# plt.plot(carrier_sin_samples_index, carrier_sin_samples, "-g", carrier_sin_samples_index, carrier_cos_samples, "-b")
# plt.show()

# Scale the samples of the input signal so that they are in the range from -1 to 1
# Масштабируем входной сигнал, что бы максимум был 1 или -1
input_signal_maximum_amplitude = max(abs(input_signal_data))
input_signal_data = input_signal_data / input_signal_maximum_amplitude

# Current sample of the input signal
# Текущий отсчёт входного сигнала
current_sample_input_signal = 0

# Envelope filter, first and second
# Для фильтрации огибающей
envelope_average_buffer_size = bit_samplerate_period_per_bit
envelope_average_buffer_counter = 0
envelope_average_buffer_0 = np.linspace(0, 0, int(envelope_average_buffer_size))
envelope_average_buffer_1 = np.linspace(0, 0, int(envelope_average_buffer_size))
envelope_average_value_0 = 0
envelope_average_value_1 = 0

# Carrier capture flag
# Флаг захвата несущей
envelope_carrier_lock_flag = 0

# Carrier lock amplitude threshold
# Порог определения захвата несущей
envelope_carrier_lock_threshold = 0.2

# Carrier capture counter and counter threshold
# Счётчик для определения захвата несущей и порог счётчика
envelope_carrier_lock_counter = 0
envelope_carrier_lock_counter_threshold = carrier_bit_size // 2

# Average value of the envelope at the time of carrier lock
# Среднее значение огибающей на момент захвата несущей
envelope_mean_value = 0

# To implement the Early-Late algorithm
# Для алгоритма Early-Late
bit_signal_el_value_hold = 0  # t
bit_signal_el_value_early = 0 # t + delta t
bit_signal_el_value_late = 0  # t - delta t
bit_signal_el_value_error = 0
bit_signal_el_time_hold = round(bit_samplerate_period_per_bit / 2)
bit_signal_el_time_early = round(bit_signal_el_time_hold - bit_samplerate_period_per_bit / 4)
bit_signal_el_time_late = round(bit_signal_el_time_hold + bit_samplerate_period_per_bit / 4)

# Envelope filter counter number in which the middle of the bit is located
# Номер счётчика фильтра огибающей в котором находиться середина бита
envelope_filter_counter_value_middle_bit = 0

# The value of the received bit and the previous bit value
# Значение принятого бита и предыдущего значения бита
bit_digital_value = 0
bit_digital_previous_value = 0

# Debug
# Для отладки
envelope_average_value_0_debug = []
envelope_average_value_1_debug = []
envelope_carrier_lock_sample_debug = []
envelope_carrier_lock_value_debug = []
envelope_mean_value_debug = []
bit_signal_el_error_value_debug = []
bit_digital_value_debug_x = []
bit_digital_value_debug_y = []
bit_digital_value_debug = []

# Loop through input samples
# Проход по входным отсчётам
for i in range(int(input_signal_length)):
    # Реализуем детектор огибающей сигнала
    # Implement the signal envelope detector

    # Get current sample input signal
    # Получим текущее значение сигнала
    current_sample_input_signal = input_signal_data[i]

    # The absolute value of the sample
    # Абсолютное значение отсёта
    current_sample_input_signal = abs(current_sample_input_signal)

    # Put the result in the first average buffer
    # Помещаем результат в первый буфер плавающего среднего
    envelope_average_buffer_0[envelope_average_buffer_counter] = current_sample_input_signal

    # Calculate the value of the floating average
    # Считаем значение плавающего среднего
    envelope_average_value_0 = np.mean(envelope_average_buffer_0)

    # Put the result in the second average buffer
    # Помещаем результат в второй буфер плавающего среднего
    envelope_average_buffer_1[envelope_average_buffer_counter] = envelope_average_value_0

    # Calculate the value of the floating average
    # Считаем значение плавающего среднего
    envelope_average_value_1 = np.mean(envelope_average_buffer_1)

    # Debug
    envelope_average_value_0_debug.append(envelope_average_value_0)
    envelope_average_value_1_debug.append(envelope_average_value_1)
    envelope_mean_value_debug.append(envelope_mean_value)
    bit_signal_el_error_value_debug.append(bit_signal_el_value_error)

    # Floating average buffer counter increment
    # Инкремент счётчика буфера плавающего среднего
    envelope_average_buffer_counter += 1
    if envelope_average_buffer_counter >= envelope_average_buffer_size:
        envelope_average_buffer_counter = 0

    # If the carrier capture flag is not set
    # Если не выставлен флаг захвата несущей
    if envelope_carrier_lock_flag == 0:
        # Carrier lock control
        # Контроль захвата несущей
        if envelope_average_buffer_counter == 0:
            # Increment the counter when the value is greater than the threshold, otherwise reset the counter
            # Инкремент счётчика когда значение больше порога, иначе сброс счётчика
            if envelope_average_value_1 > envelope_carrier_lock_threshold:
                envelope_carrier_lock_counter += 1
            else:
                envelope_carrier_lock_counter = 0

            # Checking the counter value
            # Проверка значения счётчика
            if envelope_carrier_lock_counter > envelope_carrier_lock_counter_threshold:
                # There is a carrier lock
                # Есть захват несущей
                envelope_carrier_lock_flag = 1
                envelope_carrier_lock_sample_debug.append(i)
                envelope_carrier_lock_value_debug.append(envelope_average_value_1)
                envelope_mean_value = envelope_average_value_1 / 2

    # If the carrier lock flag is set
    # Если выставлен флаг захвата несущей
    if envelope_carrier_lock_flag == 1:
        # Implementation of the Early-Late algorithm if the carrier is captured
        # Алгоритм Early-Late, после того как захвачена несущая
        # Shift values
        # early - earlier envelope value
        # hold - the middle of the envelope
        # late - late envelope value
        # Продвигаем значения
        # early - ранее значение огибающей
        # hold - середина огибающей
        # late - позднее значение огибающей
        if round(bit_signal_el_time_late) == envelope_average_buffer_counter:
            bit_signal_el_value_late = abs(envelope_average_value_1 - envelope_mean_value)
        if round(bit_signal_el_time_early) == envelope_average_buffer_counter:
            bit_signal_el_value_early = abs(envelope_average_value_1 - envelope_mean_value)
        if round(bit_signal_el_time_hold) == envelope_average_buffer_counter:
            bit_signal_el_value_hold = abs(envelope_average_value_1 - envelope_mean_value)

            # Error signal
            # Сигнал ошибки
            bit_signal_el_value_error = (bit_signal_el_value_early - bit_signal_el_value_late) / 2

            # Update the time value of the middle bit
            # Обновляем время середины бита
            bit_signal_el_time_hold -= bit_signal_el_value_error
            if bit_signal_el_time_hold < 0:
                bit_signal_el_time_hold += bit_samplerate_period_per_bit
            if bit_signal_el_time_hold > bit_samplerate_period_per_bit:
                bit_signal_el_time_hold -= bit_samplerate_period_per_bit

            # Update the time value of the late
            # Обновляем значение времени late
            bit_signal_el_time_late = bit_signal_el_time_hold + (bit_samplerate_period_per_bit / 4)
            if bit_signal_el_time_late < 0:
                bit_signal_el_time_late += bit_samplerate_period_per_bit
            if bit_signal_el_time_late > bit_samplerate_period_per_bit:
                bit_signal_el_time_late -= bit_samplerate_period_per_bit

            # Update the time value of the early
            # Обновляем значение времени early
            bit_signal_el_time_early = bit_signal_el_time_hold - (bit_samplerate_period_per_bit / 4)
            if bit_signal_el_time_early < 0:
                bit_signal_el_time_early += bit_samplerate_period_per_bit
            if bit_signal_el_time_early > bit_samplerate_period_per_bit:
                bit_signal_el_time_early -= bit_samplerate_period_per_bit

            # Continuously update the mid-bit time value
            # Постоянно обновляем значение середины бита
            if bit_signal_el_value_error == 0:
                envelope_filter_counter_value_middle_bit = round(bit_signal_el_time_hold)

        # We look at the value of the bit at the time of hold
        # Смотрим значение бита в момент hold
        if envelope_average_buffer_counter == envelope_filter_counter_value_middle_bit:
            # The value of the bit depends on the sign of the difference between the envelope and the average value
            # Значение бита зависит от знака разности огибающей и среднего значения
            if envelope_average_value_1 - envelope_mean_value <= 0:
                bit_digital_value = 0
                bit_digital_value_debug.append('0')
            else:
                bit_digital_value = 1
                bit_digital_value_debug.append('1')

            # Debug
            bit_digital_value_debug_x.append(i)
            bit_digital_value_debug_y.append(envelope_average_value_1)

    # If the carrier is captured, then we are waiting for the preamble
    # Если несущая захвачена, то ждём преамбулу
    if (costas_loop_carrier_lock_flag == 1) and (preamble_lock_flag == 0):
        # Checking for bit alternation, this is the preamble
        # Проверка на чередование бит

# Debug
# Для отладки
plt.figure("Time Во времени")
plt.plot(envelope_average_value_0_debug, "-b", envelope_average_value_1_debug, "-g", envelope_mean_value_debug, "-r", bit_signal_el_error_value_debug, "-m")
plt.title('Receive ASK, rate (приём ASK сигнала со скоростью) {0} бит/сек'.format(transmit_bitrate))
plt.xlabel('Sample Номер отсчёта', color='gray')
plt.ylabel('Filter output Выход фильтра', color='gray')

plt.plot(envelope_carrier_lock_sample_debug, envelope_carrier_lock_value_debug, 'rs')
for i in range(len(envelope_carrier_lock_sample_debug)):
    plt.annotate('carrier lock', (envelope_carrier_lock_sample_debug[i], envelope_carrier_lock_value_debug[i]), ha='center')

plt.plot(bit_digital_value_debug_x, bit_digital_value_debug_y, 'ro')
for i in range(len(bit_digital_value_debug_x)):
    plt.annotate(bit_digital_value_debug[i], (bit_digital_value_debug_x[i], bit_digital_value_debug_y[i]), ha='center')

plt.grid()
plt.show()

