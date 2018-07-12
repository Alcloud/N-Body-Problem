import numpy as np
import asyncio
import websockets
import pyopencl as cl
import pyopencl.array as cl_array

# scale factors to normalize one step to 1/10 day (3650 steps - one year)
mass_cale = 18981.3
vel_scale = 365.25

def load_ephemeris():
    # load bodies' data taken from NASA telenet
    ephemeris = np.genfromtxt('ephemeris.csv', delimiter=',', skip_header=1, dtype=np.float32, usecols=(1,3,4,5,6,7,8))
    # scale and reshape data
    mass = ephemeris[:,0]/mass_cale
    pos = np.concatenate((ephemeris[:,1:4], mass.reshape(mass.shape[0],1)), axis=1)
    vel = np.zeros((ephemeris.shape[0], 4), dtype=np.float32)
    vel[:,:3] = ephemeris[:,4:]*vel_scale
    
    return pos, vel

def start_websocket(websocket, path):
    try:
        # initialize OpenCL
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        # build kernal from file
        prg = cl.Program(ctx, open('kernal.cl').read()).build()

        # load solar system data into numpuy arrays on the host
        pos_host, vel_host = load_ephemeris()
        # create and initialize array on the device
        # copy initial postiton to device
        pos_dev = cl_array.to_device(queue, pos_host)
        # copy initial velocity to device
        vel_dev = cl_array.to_device(queue, vel_host)
        # allocate memory for new data on the devie
        pos_new_dev = cl_array.empty_like(pos_dev)
        vel_new_dev = cl_array.empty_like(vel_dev)

        # main loop
        while True:
            # run kernal with the work goup size of a number of bodies
            prg.nbody_simple(queue, (pos_host.shape[0],), None, pos_dev.data, vel_dev.data, pos_new_dev.data, vel_new_dev.data)
            # copy new position into host np.array (device -> host)
            pos_host = pos_new_dev.get()
            # update position and velocity on the device (device -> device)
            cl.enqueue_copy(queue, pos_dev.data, pos_new_dev.data)
            cl.enqueue_copy(queue, vel_dev.data, vel_new_dev.data)
            #send data to client
            yield from websocket.send(','.join(map(str,pos_host[:, :3].flatten())))
    finally:
      yield from websocket.close()


print('initializeing and starting websocket server')
start_server = websockets.serve(start_websocket, '0.0.0.0', 8778)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
