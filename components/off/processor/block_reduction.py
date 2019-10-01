#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import multiprocessing

import numpy as np


class BlockReducer(object):

    def __init__(self, n_rows, n_cols):

        self.rows = n_rows
        self.cols = n_cols
        self.ready, self.virtual, self.axes = generate_tables(n_rows, n_cols)

        self.last_scale = len(self.axes) - 1

        self.identifiers = {}

    def register_identifier(self, scale, position, identifier):

        row, col = position
        self.ready[scale][row, col] = True
        self.identifiers[(scale, row, col)] = identifier

    def get_reduce_operation(self, scale, position, identifier):

        redu_axis = self.axes[scale]
        row, col = position
        partner_position = get_scale_partner(row, col, redu_axis)
        row_p, col_p = partner_position
        partner_virtual = self.virtual[scale][row_p, col_p]
        partner_ready = self.ready[scale][row_p, col_p]

        if not partner_ready:
            target_position = None
            partner_identifier = None
        else:
            target_position = get_next_scale_position(row, col, redu_axis)
            partner_identifier = (None if partner_virtual else
                                  self.identifiers[(scale, row_p, col_p)])

        return target_position, partner_position, partner_virtual, partner_identifier

    def done(self):

        return self.ready[self.last_scale][0, 0]

    def result_identifier(self):

        return self.identifiers[(self.last_scale, 0, 0)]


def reduce_operation(identifier, partner_identifier, partner_virtual,
                     target_position, scale, reduction_axis, function):

    if partner_virtual:
        success = True
        result_identifier = identifier
    else:
        row_target, col_target = target_position
        success, result_identifier = function(identifier, partner_identifier,
                                              row_target, col_target, scale,
                                              reduction_axis)

    return success, result_identifier


class Task(object):

    def __init__(self, redu_fun, spec):

        self.redu_fun = redu_fun
        self.args = spec

    def __call__(self):

        return process_task(self.redu_fun, self.args)

    def __str__(self):

        names = ("scale", "reduction_axis", "position", "partner_position",
                 "target_position", "identifier", "partner_identifier",
                 "partner_virtual")
        return str(dict(zip(names, self.args)))


def process_task(redu_fun, task):

    (scale, reduction_axis, position, partner_position, target_position,
     identifier, partner_identifier, partner_virtual) = task

    success, result_id = reduce_operation(identifier, partner_identifier,
                                          partner_virtual, target_position,
                                          scale, reduction_axis, redu_fun)
    if success:
        result = (scale + 1, target_position, result_id)
    else:
        result = None

    return success, result


def enqueue_task(task_queue, block_reducer, redu_fun,
                 scale, position, identifier):

    op_spec = block_reducer.get_reduce_operation(scale, position, identifier)
    target_position, partner_position, partner_virtual, partner_identifier = op_spec

    if target_position is not None:
        args = (scale, block_reducer.axes[scale], position, partner_position,
                target_position, identifier, partner_identifier, partner_virtual)
        task_queue.put(Task(redu_fun, args))


def mp_process_blockreduce(block_reducer, id_array, redu_fun, max_threads):

    manager = multiprocessing.Manager()
    task_q = manager.JoinableQueue()
    result_q = manager.Queue()

    workers = [ Worker(task_q, result_q) for _ in range(max_threads) ]
    [w.start() for w in workers]

    for r in range(block_reducer.rows):
        for c in range(block_reducer.cols):
            position = (r, c)
            identifier = id_array[r][c]
            block_reducer.register_identifier(0, position, identifier)
            enqueue_task(task_q, block_reducer, redu_fun, 0, position, identifier)

    while True:
        scale, position, identifier = result_q.get()
        block_reducer.register_identifier(scale, position, identifier)

        if scale < block_reducer.last_scale:
            enqueue_task(task_q, block_reducer, redu_fun,
                         scale, position, identifier)
        else:
            [task_q.put(None) for _ in range(max_threads)]
            task_q.join()
            break


class Worker(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):

        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):

        proc_name = self.name
        while True:
            next_task = self.task_queue.get()

            if next_task is None:
                # Poison pill means shutdown
                #print("{}: Exiting".format(proc_name))
                self.task_queue.task_done()
                break

            #print("{}: {}".format(proc_name, next_task))

            success, result = next_task()
            self.task_queue.task_done()

            if success:
                self.result_queue.put(result)
            else:
                #TODO handle success = False
                print("{}: Unsuccessful job {}".format(proc_name, next_task))

        return


def serial_process_blockreduce(block_reducer, id_array, redu_fun):

    from queue import Queue

    task_q = Queue()

    for r in range(block_reducer.rows):
        for c in range(block_reducer.cols):
            position = (r, c)
            identifier = id_array[r][c]
            block_reducer.register_identifier(0, position, identifier)
            enqueue_task(task_q, block_reducer, redu_fun, 0, position, identifier)

    while not task_q.empty():
        task = task_q.get()
        success, result = task()

        if success:
            scale, position, identifier = result
            block_reducer.register_identifier(scale, position, identifier)
            if scale < block_reducer.last_scale:
                enqueue_task(task_q, block_reducer, redu_fun,
                             scale, position, identifier)
            else:
                print(identifier)

        #TODO handle success = False


def generate_tables(n_rows, n_cols):

    dimensions_raw = np.r_[n_rows, n_cols]
    dimensions = 2**(np.ceil(np.log2(dimensions_raw)).astype(int))

    axes = []
    is_virtual = []
    is_ready = []

    imgs_are_virtual = np.ones(dimensions, dtype=bool)
    imgs_are_virtual[0:dimensions_raw[0], 0:dimensions_raw[1]] = False
    is_virtual.append(imgs_are_virtual)

    reals_are_ready = np.zeros(dimensions, dtype=bool)
    reals_are_ready[imgs_are_virtual] = True
    is_ready.append(reals_are_ready)

    while dimensions.prod() > 1:

        reduction_axis = (dimensions.argmax() if
                          (dimensions.max() != dimensions.min()) else 0)

        dimensions[reduction_axis] /= 2
        this_virtual = populate_virtual(dimensions, is_virtual[-1],
                                        reduction_axis)
        this_ready = np.zeros(dimensions, dtype=bool)
        this_ready[this_virtual] = True

        axes.append(reduction_axis)
        is_virtual.append(this_virtual)
        is_ready.append(this_ready)

    # Last scale is irreducible
    axes.append(-1)

    return is_ready, is_virtual, axes


def populate_virtual(this_scale_shape, previous_scale, reduction_axis):

    this_scale = np.zeros(this_scale_shape, dtype=bool)
    for row, col in np.ndindex(*this_scale_shape):
        this_scale[row, col] = scale_step_value(row, col, previous_scale,
                                                reduction_axis)
    return this_scale


def scale_step_value(row, col, previous_scale, reduction_axis):

    t_row = row if reduction_axis == 1 else 2*row
    o_row = 0 if reduction_axis == 1 else 1
    t_col = col if reduction_axis == 0 else 2*col
    o_col = 0 if reduction_axis == 0 else 1
    value = (previous_scale[t_row, t_col] and
             previous_scale[t_row+o_row, t_col+o_col])
    return value


def get_scale_partner(row, col, reduction_axis):

    if reduction_axis == 0:
        col_p = col
        row_p = row + 1 if (row % 2 == 0) else row - 1
    elif reduction_axis == 1:
        row_p = row
        col_p = col + 1 if (col % 2 == 0) else col - 1

    return row_p, col_p


def get_next_scale_position(row, col, reduction_axis):

    row_p, col_p = get_scale_partner(row, col, reduction_axis)

    if reduction_axis == 0:
        col_n = col
        row_n = int(min(row, row_p)/2)
    elif reduction_axis == 1:
        row_n = row
        col_n = int(min(col, col_p)/2)

    return row_n, col_n


def test_main():

    n_rows = int(sys.argv[1])
    n_cols = int(sys.argv[2])

    ready, virtual, axes = generate_tables(n_rows, n_cols)
    last_scale = len(axes) - 1
    axis_name = ['rows', 'cols']

    for scale, (_ready, _virtual) in enumerate(zip(ready, virtual)):

        redu_axis = axes[scale]
        redu_msg = ("is irreducible" if (redu_axis == -1) else
                    "to reduce over {}".format(axis_name[redu_axis]))

        print("{}\nScale {} {}".format("- "*40, scale, redu_msg))
        print("Virtual")
        print(_virtual)
        print("Ready")
        print(_ready)
        print('\n')

        if scale < last_scale:
            print("Reduction Partners to scale {}".format(scale + 1))

            for row, col in np.ndindex(*_ready.shape):
                row_p, col_p = get_scale_partner(row, col, redu_axis)
                print("   ({}, {}) -> ({}, {})".format(row, col, row_p, col_p))

            print("Target position on scale {}".format(scale + 1))

            for row, col in np.ndindex(*_ready.shape):
                row_n, col_n = get_next_scale_position(row, col, redu_axis)
                print("   ({}, {}) -> ({}, {})".format(row, col, row_n, col_n))

            print()

    print("\n{}\nTesting BlockReducer:".format("="*80))
    reducer = BlockReducer(n_rows, n_cols)

    #fun = lambda x, y, rt, ct: (True, x + y)
    fun = CallablePickableExample()

    id_fun = lambda r, c: r + c
    mat = [ [ id_fun(r, c) for c in range(n_cols) ] for r in range(n_rows) ]
    target = sum( sum(r) for r in mat )

    #serial_process_blockreduce(reducer, mat, fun)
    mp_process_blockreduce(reducer, mat, fun, 6)

    print("Reduction target: {}".format(target))
    if reducer.done():
        print("Reducer done: {}".format(reducer.result_identifier()))
    else:
        print("Reducer not done!")
        pr = ["{}: {}".format(key, reducer.identifiers[key])
              for key in sorted(reducer.identifiers)]
        print('\n'.join(pr))


class CallablePickableExample(object):

    def __call__(self, id_a, id_b, row, col, scale, redu_axis):
        #time.sleep(10)
        return True, id_a + id_b


if __name__ == "__main__":
    import sys
    import time
    test_main()
