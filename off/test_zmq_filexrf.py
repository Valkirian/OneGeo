#!/usr/bin/env python2

import sys
from threading import Thread

import zmq
from zhelpers import zpipe

CHUNK_SIZE = 32*1024
CHUNKED = False

def client_thread(ctx, pipe):
    receiver = ctx.socket(zmq.PULL)
    receiver.connect("tcp://127.0.0.1:6000")

    if CHUNKED:
        chunks = 0      # Total chunks received
        rec_buff = []

        while True:
            try:
                chunk = receiver.recv()
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    return   # shutting down, quit
                else:
                    raise

            chunks += 1
            rec_buff.append(chunk)
            if len(chunk) == 0:
                break   # whole file received

        buffer = b''.join(rec_buff)
        print ("%i chunks received, %i bytes" % (chunks, len(buffer)))

    else:
        buffer = receiver.recv()
        print ("single chunk received, %i bytes" % (len(buffer)))

    pipe.send(b"OK")

    with open('/dev/shm/test.out', 'wb') as fout:
        fout.write(buffer)

# The server thread reads the file from disk in chunks, and sends
# each chunk to the client as a separate message. We only have one
# test file, so open that once and then serve it out as needed:

def server_thread(ctx):
    file = open(sys.argv[1], "r")

    sender = ctx.socket(zmq.PUSH)

    if CHUNKED:
        # Default HWM is 1000, which will drop messages here
        # since we send more than 1,000 chunks of test data,
        # so set an infinite HWM as a simple, stupid solution:
        sender.setsockopt(zmq.SNDHWM, 0)

    sender.bind("tcp://*:6000")

    while True:

        if CHUNKED:
            while True:
                data = file.read(CHUNK_SIZE)
                sender.send_multipart(data)
                if not data:
                    break
        else:
            data = file.read()
            sender.send(data)
            break

# The main task starts the client and server threads; it's easier
# to test this as a single process with threads, than as multiple
# processes:

def main():

    # Start child threads
    ctx = zmq.Context()
    a,b = zpipe(ctx)

    client = Thread(target=client_thread, args=(ctx, b))
    server = Thread(target=server_thread, args=(ctx,))
    client.start()
    server.start()

    # loop until client tells us it's done
    try:
        print a.recv()
    except KeyboardInterrupt:
        pass
    del a,b
    ctx.term()

if __name__ == '__main__':
    main()
