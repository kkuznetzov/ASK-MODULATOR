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

# Input data file to transfer
# Имя файла с данными
data_file_name = 'transmit_data.txt'
data_file_name = os.path.join(os.path.dirname(__file__), data_file_name)

# Output audio file, modulated signal
# Имя выходного wav файла
wav_file_name = 'wav\\ask_out.wav'
wav_file_name = os.path.join(os.path.dirname(__file__), wav_file_name)

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

# Number of wav sampling samples per data bit
# Число отсчётов частоты дискретизации на каждый бит
bit_samplerate_period_per_bit = wav_samplerate / transmit_bitrate

# Open txt file for reading
# Открываем файл на чтение
data_file = open(data_file_name, "rb")

# Reading a file
# Читаем файл
input_signal_data = bytearray(data_file.read())
input_signal_length_bytes = len(input_signal_data)
input_signal_length_bits = input_signal_length_bytes * 8

# Data stream size with preamble, sync word and postamble, bits
# Размер потока данных с преамбулой,  словом синхронизации и постамбулой, бит
output_signal_data_bits = input_signal_length_bits + carrier_bit_size + preamble_bit_size + synchronization_word_bit_size + postamble_bit_size

# Calculate the duration of all data in wav file samples
# Считаем длительность посылки в отсчётах частоты дискретизации
output_signal_sample_count = output_signal_data_bits * bit_samplerate_period_per_bit

# Output parameters to the console
# Вывод параметров
print("Частота дискретизации wav =", wav_samplerate)
print("Частота несущей =", carrier_frequency)
print("Число периодов дискрертизации на бит =", bit_samplerate_period_per_bit);
print("Размер файла входных данных, байт =", input_signal_length_bytes)
print("Размер файла входных данных, бит =", input_signal_length_bits)
print("Размер несущей, бит =", carrier_bit_size)
print("Амплитуда несущей, бит =", carrier_bit_value)
print("Размер преамбулы, бит =", preamble_bit_size)
print("Размер постамбулы, бит =", postamble_bit_size)
print("Размер слова синхронизации, бит =", synchronization_word_bit_size)
print("Значение бит слова синхронизации =", synchronization_word_bit_value)
print("Размер выходных данных с преамбулой и словом синхронизации, бит =", output_signal_data_bits)
print("Число отсчётов выходного wav файла =", output_signal_sample_count)

# Calculate an array with sinusoid samples
# Формируем отсчёты синусоиды
carrier_sin_samples_index = np.arange(bit_samplerate_period_per_bit)
carrier_sin_samples = np.sin(2 * np.pi * (carrier_frequency / transmit_bitrate) * carrier_sin_samples_index / bit_samplerate_period_per_bit)

# Empty array for signal
# Формруем выходные данные, пока всё 0
output_signal = np.linspace(0, 0, int(output_signal_sample_count))

# Byte counter and bit counter for input file
# Счётчик байт и счётчик бит
input_byte_count = 0
input_bit_count = 0
output_bit_count = 0

# Byte value and bit value
# Значение байта и значение бита
byte_value = 0
bit_value = 0

# Phase counter. Sine sample Counter
# Счётчик фазы
phase_cnt = 0

# Амплитууда сигнала
signal_amplitude = 0

# Creating a ASK signal
# Формируем выходной сигнал согласно битам, используя ASK
for i in range(int(output_signal_sample_count)):
    # In the beginning, we make samples of the carrier frequency
    # Формируем несущую
    if output_bit_count < carrier_bit_size:
        if carrier_bit_value == 0:
            bit_value = 0
        else:
            bit_value = 1

    # Make preamble, alternating frequency
    # Формируем биты преамбулы
    if (output_bit_count >= carrier_bit_size) and (output_bit_count < carrier_bit_size + preamble_bit_size):
        # Signal frequency alternating
        # Значение бита преамбулы, инверсия когда счётчик фазы равен 0
        if phase_cnt == 0:
            if bit_value == 0:
                bit_value = 1
            else:
                bit_value = 0

    # Make sync word
    # Биты слова синхронизации
    if (output_bit_count >= carrier_bit_size + preamble_bit_size) and (output_bit_count < (carrier_bit_size + preamble_bit_size + synchronization_word_bit_size)):
        if phase_cnt == 0:
            bit_value = synchronization_word_bit_value

    # Read input data bytes/bits
    # Биты входных данных
    if (output_bit_count >= (carrier_bit_size + preamble_bit_size + synchronization_word_bit_size)) and (output_bit_count < (carrier_bit_size + preamble_bit_size + synchronization_word_bit_size + input_signal_length_bits)):
        # Read one byte when phase counter is 0
        # Байт из входного файла читаем когда счётчик фазы равен 0
        if phase_cnt == 0:
            byte_value = input_signal_data[input_byte_count]

            # Get single bit from byte
            # Бит из байта
            if (byte_value >> input_bit_count) & 1 == 0:
                bit_value = 0
            else:
                bit_value = 1

            # Счётчики бит и байт
            input_bit_count += 1
            if input_bit_count == 8:
                input_bit_count = 0
                input_byte_count += 1

    # Postamble
    # Постамбула
    if output_bit_count >= (carrier_bit_size + preamble_bit_size + synchronization_word_bit_size + input_signal_length_bits):
        # Signal frequency inversion
        # Значение постамбулы, инверсия когда счётчик фазы равен 0
        if phase_cnt == 0:
            if bit_value == 0:
                bit_value = 1
            else:
                bit_value = 0

    # Set the amplitude of the signal
    # Задаём амплитуду сигнала
    if bit_value == 0:
        output_signal[i] = carrier_sin_samples[phase_cnt] * bit_0_amplitude
    else:
        output_signal[i] = carrier_sin_samples[phase_cnt] * bit_1_amplitude

    # Increment phase counter
    # Счётчик фазы
    phase_cnt += 1
    if phase_cnt >= bit_samplerate_period_per_bit:
        phase_cnt = 0
        output_bit_count += 1

# Save wav file
# Сохраним в файл
output_signal *= 32765
output_signal_int = np.int16(output_signal)
wavfile.write(wav_file_name, wav_samplerate, output_signal_int)