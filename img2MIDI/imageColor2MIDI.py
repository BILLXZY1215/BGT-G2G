import cv2
from PIL import Image
import numpy as np
import pretty_midi
import sys
import webcolors
import math
from collections import Counter
import matplotlib.pyplot as plt

colormap = {
    # (R, G, B)
    # Resource:
    # https://www.color-name.com/
    # https://www.w3.org/wiki/CSS/Properties/color/keywords
    'C': (255, 0, 0),  # red
    'F': (130, 0, 0),  # dark red
    'G': (255, 165, 0),  # orange
    'D': (255, 255, 0),  # yellow
    'A': (0, 255, 0),  # green
    'E': (135, 206, 235),  # sky blue
    'B': (0, 0, 255),  # blue
    'F#': (24, 62, 250),  # bright blue
    'C#': (238, 130, 238),  # violet
    'G#': (200, 162, 200),  # lilac
    'D#': (210, 169, 161),  # flesh
    'A#': (255, 228, 225),  # mistyrose
}

C_Major = {
    'C': 0,  # C5
    'C#': 1,
    'D': 2,
    'D#': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'G': 7,
    'G#': 8,
    'A': 9,
    'A#': 10,
    'B': 11,
}

chord_progress = {
    # Example in C Major
    '1645': [0, 9, 5, 7],  # C Am F G (Folk)
    '1564': [0, 7, 9, 5],  # C G Am F (Pop-Punk Progression)
    '1563': [0, 7, 9, 4],  # C G Am Em (My original song progressive)
    '6415': [0, -4, -9, -2],  # Am F C G
    '15634125': [0, 7, 9, 4, 5, 0, 2, 7],  # C G Am Em F C Dm G (Canon)
    '4536251': [0, 2, -1, 4, -3, 2, -5],  # F G Em Am Dm G C
    '1526415': [0, 7, 2, 9, 5, 0, 7],  # C G Dm Am F C G
    '2514736': [0, 5, -2, 3, 9, 2, 7],  # Dm G C F Bm Em Am
    '6251423': [0, -7, -2, -9, -4, -7, -5], # Am Dm G C F D7 E7
}


def three_distance(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)


def findClosest(rgb):
    # rgb: (R, G, B)
    min_distance = sys.maxsize
    res = ''
    for note in colormap:
        delta_R = rgb[0] - colormap[note][0]
        delta_G = rgb[1] - colormap[note][1]
        delta_B = rgb[2] - colormap[note][2]
        dist = three_distance(delta_R, delta_G, delta_B)
        if (dist < min_distance):
            min_distance = dist
            res = note
    return res


def noteName2Value(note_name):
    return C_Major[note_name]


def melody(note_list, chord_progress_type):
    root_list = [i[0] for i in note_list]
    # Apply 1645  root: (n, n+9, n+5, n+7)
    temp_list = []
    for i in range(1, len(note_list)):
        # Find out consecutive root note, which has same note values
        if root_list[i] == root_list[i-1]:
            temp_list.append(i-1)
        else:
            temp_list.append(i-1)
            root = root_list[temp_list[0]]
            k = 0
            for item in temp_list:
                chord_progress_length = len(
                    chord_progress[chord_progress_type])
                for x in range(1, chord_progress_length):
                    if k % chord_progress_length == x:
                        root_list[item] = root_list[item] + \
                            chord_progress[chord_progress_type][x]
                        note_list[item] = [
                            i + chord_progress[chord_progress_type][x] for i in note_list[item]]
                k += 1
            temp_list = []
    return note_list


# Octave Ranges
# -1: 21-23 (A0, A#, B0)
# 0: 24-35 (C1-B1)
# 1: 36-47 (C2-B2)
# 2: 48-59 (C3-B3)
# 3: 60-71 (C4-B4)
# 4: 72-83 (C5-B5)
# 5: 84-95 (C6-B6)
# 6: 96-107 (C7-B7)
# 7: 108-119 (C8-B8)
# 8: 120-127 (C9-G9)


def note2octave(notes):
    octaves = []
    for note in notes:
        octaves.append(int(note/12)-2)
    return octaves


def imageColor2MIDI(image_path, interval, chord_progress_type):
    interval = float(interval)
    c_chord = pretty_midi.PrettyMIDI()
    EGP = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program('Electric Grand Piano'))
    img = Image.open(image_path)  # RGB Matrix
    img = img.resize((64, 64))  # resize to a portable size
    if img.mode == 'RGB':
        img = list(img.convert('RGB').getdata())
        pixels = [(item, img.count(item)) for item in set(img)]
        # sort by frequency (High Frequency RGB values appear first)
        pixels = sorted(pixels, key=lambda x: x[1], reverse=True)
        # Combine Piano Row Matrix in Color information
        img2 = np.array(Image.open(image_path))  # RGB Matrix
        img2 = cv2.resize(img2, (len(pixels), 88))
        img2 = np.dot(img2, [0.33, 0.33, 0.33])
        # Find octave of the (most frequent and pixel-max) pixel value
        octaves = []
        for piano_row in img2.T:
            unique_array = np.unique(piano_row, return_counts=True)
            value_unique_array = unique_array[0]
            count_unique_array = unique_array[1]
            temp = np.sort(count_unique_array)
            # Get the most frequent and pixel-max pixel value
            max_freq_pixel = max(value_unique_array[np.where(
                count_unique_array == temp[-1])])
            max_freq_index = np.where(
                piano_row == max_freq_pixel)
            # Find the middle index of max_freq_index (+ 21)
            octaves.append(max_freq_index[0]
                           [int(len(max_freq_index[0])/2)] + 21)
        octaves = note2octave(octaves)
        notes = []
        k = 0
        for color, _ in pixels:
            rgb = (color[0], color[1], color[2])
            notes.append(12*(octaves[k] + 2) +
                         noteName2Value(findClosest(rgb)))  # octave * 12 => note C, + offset
            k += 1
        notes = [[note, note+3, note+7]
                 for note in notes]  # Default (0, +4, +7) Major Triad
        notes = melody(notes, chord_progress_type)
        root_notes = [note[0] for note in notes]
        # print(root_notes)
        n, bins, patches = plt.hist(
            # bins: number of bars in histogram
            # alpha: opacity
            root_notes, bins=max(root_notes)-min(root_notes), facecolor='green', alpha=0.75)
        plt.savefig(
            './histogram/Histogram-RootNote-{}.jpg'.format(image_path.split('.')[0]))
        i = 0
        for items in notes:
            for item in items:
                note = pretty_midi.Note(
                    velocity=100, pitch=item, start=0+interval*i, end=interval*(i+1))
                EGP.notes.append(note)
            i = i + 1
        c_chord.instruments.append(EGP)
        c_chord.write('content_ICM.mid')

    else:
        print('Unimplemented image type')


image_path = sys.argv[1]
interval = sys.argv[2]
chord_progress_type = sys.argv[3]

imageColor2MIDI(image_path, interval, chord_progress_type)
