import cv2
from PIL import Image
import numpy as np
import pretty_midi
import sys

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
    '6251423': [0, -7, -2, -9, -4, -7, -5],  # Am Dm G C F D7 E7
}


def findClosest(A, B, C):
    A = A.tolist()
    B = B.tolist()
    C = C.tolist()
    p = len(A)
    q = len(B)
    r = len(C)
    # Initialize min diff
    diff = sys.maxsize
    res_i = 0
    res_j = 0
    res_k = 0
    # Traverse Array
    i = 0
    j = 0
    k = 0
    while(i < p and j < q and k < r):
        # Find minimum and maximum of
        # current three elements
        minimum = min(A[i], min(B[j], C[k]))
        maximum = max(A[i], max(B[j], C[k]))
        # Update result if current diff is
        # less than the min diff so far
        if maximum-minimum < diff:
            res_i = i
            res_j = j
            res_k = k
            diff = maximum - minimum
        # We can 't get less than 0 as
        # values are absolute
        if diff == 0:
            break
        # Increment index of array with
        # smallest value
        if A[i] == minimum:
            i = i+1
        elif B[j] == minimum:
            j = j+1
        else:
            k = k+1
        # Print result
    # print(A[res_i], ' ', B[res_j], ' ', C[res_k])
    return [A[res_i],  B[res_j], C[res_k]]

# TODO: Set Main Major


def threeChordMapping(note_list):
    if len(note_list) != 3:
        return
    # Find two nearest element index (only support three elements)
    # Default: 0 1 2 / 1 2 0
    one = 0
    two = 1
    three = 2
    diff = abs(note_list[1] - note_list[0])
    if(abs(note_list[2] - note_list[0]) < diff):
        # 0 2 1 / 2 0 1
        two = 2
        three = 1
        diff = abs(note_list[2] - note_list[0])
    if(abs(note_list[2] - note_list[1]) < diff):
        # 1 2 0 / 2 1 0
        one = 1
        two = 2
        three = 0
        diff = abs(note_list[2] - note_list[1])

    # print('origin: ', note_list)
    # print('two nearest: ', note_list[one], note_list[two])
    while(diff > 12):  # If the minimal diff is still > 12, set note to same range
        if(note_list[one] > note_list[two]):
            note_list[two] = note_list[two] + 12
        else:
            note_list[one] = note_list[one] + 12
        diff = abs(note_list[one] - note_list[two])

    # Now, find the farest to the farest element
    n = one
    if(abs(note_list[two] - note_list[three]) > abs(note_list[one] - note_list[three])):
        n = two
    diff = abs(note_list[three] - note_list[n])
    while(diff > 12):
        if(note_list[three] > note_list[n]):
            note_list[three] = note_list[three] - 12
        else:
            note_list[three] = note_list[three] + 12
        diff = abs(note_list[three] - note_list[n])
    # Now, three notes should be in the same range
    # Set: (0, +3, +7), (0, +4, +7)
    chord_set = [[3, 7], [4, 7]]
    new_note_list = [note_list[one], note_list[two], note_list[three]]
    new_note_list.sort()
    # print(new_note_list)
    # Default
    snd_expect_value = new_note_list[0] + 3
    trd_expect_value = new_note_list[0] + 6
    diff = (new_note_list[1] - snd_expect_value)**2
    +(new_note_list[2] - trd_expect_value)**2
    res = [3, 6]
    # Euclidean Distance: Find the most likely chord
    for chord_offset in chord_set:
        snd_expect_value = new_note_list[0] + chord_offset[0]
        trd_expect_value = new_note_list[0] + chord_offset[1]
        temp = (new_note_list[1] - snd_expect_value)**2
        +(new_note_list[2] - trd_expect_value)**2
        if temp < diff:
            res = chord_offset
    new_note_list[1] = new_note_list[0] + res[0]
    new_note_list[2] = new_note_list[0] + res[1]
    # print(new_note_list)
    return new_note_list


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
            print("temp list: ", temp_list)
            temp_list = []

    print(root_list)
    return note_list


def image2MIDI(image_path, interval, chord_progress_type):
    interval = float(interval)
    c_chord = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a specified instrument
    # TODO: Implement Instrument Name Category
    # EGC = pretty_midi.Instrument(
    #     program=pretty_midi.instrument_name_to_program('Electric Guitar (clean)'))

    EBF = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program('Electric Bass (finger)'))

    EGP = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program('Electric Grand Piano'))

    img = np.array(Image.open(image_path))  # RGB Matrix
    img = cv2.resize(img, (100, 88))
    img = np.dot(img, [0.33, 0.33, 0.33])  # TODO: improvement on colors

    i = 0
    three_chord_indices = []
    for piano_row in img.T:
        unique_array = np.unique(piano_row, return_counts=True)
        value_unique_array = unique_array[0]
        count_unique_array = unique_array[1]
        temp = np.sort(count_unique_array)
        # Get the three most frequent and pixel-max pixel value
        max_freq_pixel = max(value_unique_array[np.where(
            count_unique_array == temp[-1])])
        snd_freq_pixel = max(value_unique_array[np.where(
            count_unique_array == temp[-2])])
        trd_freq_pixel = max(value_unique_array[np.where(
            count_unique_array == temp[-3])])
        # print('pixel: ', max_freq_pixel, snd_freq_pixel, trd_freq_pixel)
        max_freq_index = np.where(
            piano_row == max_freq_pixel)
        snd_freq_index = np.where(
            piano_row == snd_freq_pixel)
        trd_freq_index = np.where(
            piano_row == trd_freq_pixel)
        # print('index: ', max_freq_index, snd_freq_index, trd_freq_index)
        most_freq_closet_index = findClosest(
            max_freq_index[0], snd_freq_index[0], trd_freq_index[0])
        three_chord_index = threeChordMapping(most_freq_closet_index)
        # print(three_chord_index)
        three_chord_indices.append(three_chord_index)
    # TODO: chord mapping
    # Iterate over note names, which will be converted to note number later
    # Example: C Major
    # C D E F G A B C
    # 1 2 3 4 5 6 7 1
    # 72 74 76 77 79 81 83 84
    # C = ['C5', 'E5', 'G5']  # 1 3 5 (72 76 79) (0, +4, +7)
    # Dm = ['D5', 'F5', 'A5']  # 2 4 6 (74 77 81) (0, +3, +7)
    # D = ['D5', 'F5#', 'A5']  # 2 4 6 (74 78 81) (0, +4, +7)
    # Em = ['E5', 'G5', 'B5']  # 3 5 7 (76 79 83) (0, +3, +7)
    # F = ['F5', 'A5', 'C6']  # 4 6 1 (77 81 84) (0, +4, +7)
    # G = ['G5', 'B5', 'D6']  # 5 7 2 (79, 83, 86) (0, +4, +7)
    # Am = ['A5', 'C6', 'E6']  # 6 1 3 (81 84 88) (0, +3, +7)
    # Bm = ['B5', 'D6', 'F6#']  # 7 2 4 (83 86 90) (0, +3, +7)
    # Set: (0, +3, +7), (0, +4, +7)
    # TODO: Implement chord database
    # chord_progress = [C, Am, F, G]  # Sample: 1645
    three_chord_melody_indices = melody(
        three_chord_indices, chord_progress_type)
    for three_chord_index in three_chord_melody_indices:
        for note_name in three_chord_index:
            note_number = note_name + 21
            note = pretty_midi.Note(
                velocity=100, pitch=note_number, start=0+interval*i, end=interval*(i+1))
            # Bass: 40 - 400 Hz (27 -> 67) (A0 -> G4)
            # Piano: 27 - 4200 Hz (21 -> 108) (A0 -> C8)
            if note_number <= 67 and note_number >= 27:
                EBF.notes.append(note)
            EGP.notes.append(note)
        i = i + 1
    # Add the instr instrument to the PrettyMIDI object
    c_chord.instruments.append(EBF)
    c_chord.instruments.append(EGP)
    # Write out the MIDI data
    c_chord.write('content.mid')


image_path = sys.argv[1]
interval = sys.argv[2]
chord_progress_type = sys.argv[3]

image2MIDI(image_path, interval, chord_progress_type)
