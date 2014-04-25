#! /usr/bin/env python

# This is some sandpit code for examining the results of Turing style reaction diffusion equations
# 

import warnings
import threading
import time
import sys
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.special
import random
import pyaudio

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation

# This is a step-debugging library. Way useful.
import pdb

# This is a back-end that makes the code work for Mac OSX
#matplotlib.use('TkAgg')

# This is just to remove some annoying convergence warnings that can really crowd the environment
warnings.filterwarnings('ignore', category=RuntimeWarning)

class Repressilator:

    def __init__(self, alpha=216., alpha_0=0.216, beta=5., n=2.,
                 K_m=scipy.log(2.)/120., K_p=scipy.log(2.)/600.,
                 T=20., K_b=1600., buffer_length=4096, sample_rate=44100,
                 frame_buffers = 10, t_transform=1.,
                 ms_0=[0.1,0.2,0.3,0.,0.,0.]):
        self.alpha = alpha
        self.alpha_0 = alpha_0
        self.beta = beta
        self.n = n
        self.K_m = K_m
        self.K_p = K_p
        self.T = T
        self.K_b = K_b
        self.ms_0 = ms_0

        self.buffer_length = buffer_length
        self.sample_rate = sample_rate
        self.t_transform = t_transform
        
        self.sonify_select = [0, 1]
        
        self.main_buffer = np.zeros([self.buffer_length, len(self.sonify_select)], dtype='int16')
        self.main_t = np.linspace(0., self.t_transform * float(self.buffer_length) / float(self.sample_rate), 
                             self.buffer_length)
        self.is_new_main_buffer = False
         
        self.frame_buffer = np.zeros([self.buffer_length * float(frame_buffers), len(self.sonify_select)])
        self.frame_t = np.linspace(0., float(frame_buffers) * self.t_transform 
                                       * float(self.buffer_length) / float(self.sample_rate), 
                                   self.buffer_length * frame_buffers)
        self.is_new_frame_buffer = False

        self.overdrive = 2.0

    # Here we define out reaction diffusion system
    # t is time, y the state vector (corresponds to both Y_r and X_r in Turing's paper), 
    # and a the set of coefficients
    def dmdt(self, ms, t, n, alpha, alpha_0, 
             beta, T, K_m, K_b, K_p):
        
        m = ms[:len(ms)/2]
        p = ms[len(ms)/2:]

        # The vector xs consists of X_r in the first half, and Y_r in the second
        dmdt = np.zeros(len(m))
        dpdt = np.zeros(len(p))
        
        for i in range(len(m)):
            # We take a modulo to take care of the circular arrangment of protein/mRNA 
            # interaction
            j = (i - 1) % len(m)
            # These are equations from Box 1 in Elowitz and Leibler
            dmdt[i] = -m[i] + alpha / (1 + pow(p[j], n)) + alpha_0
            dpdt[i] = -beta * (p[i] + m[i])
         
        result = np.append(dmdt, dpdt) 
        return result

    def solve_buffer(self):
        if not self.is_new_main_buffer:
            # The sound thread has consumed the buffer so we make a new one
            ms = scipy.integrate.odeint(self.dmdt, self.ms_0, self.main_t, 
                                        args = (self.n, self.alpha, self.alpha_0, self.beta, 
                                        self.T, self.K_m, self.K_b, self.K_p), atol=1e-2, rtol=1e-3)
            
            #self.stream.write(ms[:, self.sonify_select].astype('float16'), exception_on_underflow=False) 
            self.ms_0 = ms[-1, :]
           
            # first we normalise the data  
            self.main_buffer = (self.overdrive * ms[:, self.sonify_select] / np.abs(ms).max(1)[:, np.newaxis]).astype(np.float32).flatten()
            #pdb.set_trace()
            print self.main_buffer.astype(np.float32)
            # then we encode it
            #self.main_buffer_frames = self.main_buffer.flatten().astype(np.int16).tostring()

            #pdb.set_trace() 
            self.frame_buffer[0:-self.buffer_length, :] = self.frame_buffer[self.buffer_length:, :]
            self.frame_buffer[-self.buffer_length:, :] = ms[:, self.sonify_select]
            
            self.is_new_main_buffer = True
            self.is_new_frame_buffer = True

        return self.main_buffer

    def poll_buffer(self, in_data, frame_count, time_info, status):
        # For now the solve routine is here. Will want to move it to its own thread later
        self.solve_buffer()
        
        if self.is_new_main_buffer:
            self.is_new_main_buffer = False
        
        return (self.main_buffer, pyaudio.paContinue)
    
    def poll_frame_buffer(self):
        if self.is_new_frame_buffer:
            self.is_new_frame_buffer = False
        return self.frame_buffer

    # animation function.  This is called sequentially
    def animate(self, i):
        # Solve the ODE from the previous start point
        #ms = self.solve(self.ms_0);
        
        line1.set_data(self.frame_t, self.frame_buffer[:, 0])
        line2.set_data(self.frame_t, self.frame_buffer[:, 1])
        #line3.set_data(self.frame_t, self.frame_buffer[:, 2])
        return line1, line2 #, line3

"""
class solveThread(threading.Thread):
    def __init__(self, repressilator):
        threading.Thread.__init__(self)
        self.repressilator = repressilator

    def run(self):
        self.repressilator.solve_buffer() 

class soundThread(threading.Thread):
    def __init__(self, repressilator):
        threading.Thread.__init__(self)
        self.repressilator = repressilator

        # Instance of the audio stream
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16,
                channels=2,
                rate=self.repressilator.sample_rate,
                output=True,
                frames_per_buffer = self.repressilator.buffer_length)

    def run(self):
        if self.repressilator.poll_buffer():
            # add buffer to sound buffer, only if new
            self.stream.write(ms[:, self.sonify_select].astype('float16'), exception_on_underflow=False) 


class animateThread(threading.Thread):
    def __init__(self, repressilator):
        threading.Thread.__init__(self)
        self.repressilator = repressilator

    def run(self):
        self.repressilator.solve_buffer() 
"""

ms_0 = [0.1, 0.2, 0.3, 0., 0., 0.]

rep = Repressilator(t_transform = 200.)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
#ax = plt.axes(xlim=(0, end_t), ylim=(min(0.0, np.min(abs(ms[:,0:3]))), np.max(abs(ms[:,0:3]))))
ax = plt.axes(xlim=(0, max(rep.frame_t)), ylim=(-1, 100))

#line1, = ax.plot(t, ms[:,0], lw=2)
#line2, = ax.plot(t, ms[:,1], lw=2)
#line3, = ax.plot(t, ms[:,2], lw=2)
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
#line3, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    #line3.set_data([], [])
    return line1, line2#, line3

# Instance of the audio stream
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paFloat32,
        channels=2,
        rate=rep.sample_rate,
        output=True,
        frames_per_buffer = rep.buffer_length,
        stream_callback=rep.poll_buffer)

stream.start_stream()

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, rep.animate, init_func=init,
                               frames=1, interval=20, blit=False)
plt.show()

while stream.is_active():
    time.sleep(1.0)
    rep.beta = 10.0

#pdb.set_trace()

