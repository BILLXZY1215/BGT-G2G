import sys
import pretty_midi
import matplotlib.pyplot as plt

midi_path = sys.argv[1]


def histogram_mid(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    i = 0
    for instrument in midi_data.instruments:
        notes = []
        for k in instrument.notes:
            notes.append(k.pitch)
        n, bins, patches = plt.hist(
            # bins: number of bars in histogram
            # alpha: opacity
            notes, bins=max(notes)-min(notes), facecolor='green', alpha=0.75, label=instrument.name)
        plt.legend()
        plt.savefig(
            './histogram/Histogram-RealSong-{}-{}.jpg'.format(midi_path.split('.')[0], i))
        plt.clf()
        i += 1


histogram_mid(midi_path)
