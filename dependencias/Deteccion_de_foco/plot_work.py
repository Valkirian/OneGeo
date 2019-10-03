import matplotlib.pyplot as plt
import json
from glob import glob
import os.path as pth

color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
plt.ion()

kinds = ('ppl', 'xpl')

roi = 1024

files = { kind: glob(pth.join(kind, str(roi), "focus-curve-*.json")) for kind in kinds }
data = { kind: { int(pth.splitext(pth.basename(fi).split('-')[-1])[0]): json.load(open(fi)) for fi in file_l } for kind, file_l in files.items() }

for kind in kinds:
    plt.figure()
    plot_gen = ( list(zip(*data)) + [i, v] for i, (v, data) in enumerate(sorted(data[kind].items())) )
    [plt.plot(x, y, '-o', label="P = {}".format(v), color=color_sequence[i]) for x, y, i, v in plot_gen ]
    plt.title("Focus({0}), {1}x{1}".format(kind, roi))
    plt.legend()

#for kind in kinds:
#    plt.figure()
#    plot_gen = ( list(zip(*data)) + [i, v] for i, (v, data) in enumerate(sorted(data[kind].items())) )
#    [plt.plot(x, [1/(_ + 1) for _ in y], '-o', label="P = {}".format(v), color=color_sequence[i]) for x, y, i, v in plot_gen ]
#    plt.title("Low Freq Norm({0}), {1}x{1}".format(kind, roi))
#    plt.legend()
