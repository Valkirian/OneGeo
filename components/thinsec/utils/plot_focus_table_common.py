import json

from scipy import interpolate


def gen_interpolator(table, spread):

    x, y, z = zip(*table)
    focus_map = interpolate.Rbf(x, y, z, function='gaussian', epsilon=spread)

    return focus_map


def get_map(table_file):

    with open(table_file, 'r') as fobj:
        config = json.load(fobj)
    table = config['table']
    spread = config['spread']
    focus_map = gen_interpolator(table, spread)

    return focus_map, table, spread
