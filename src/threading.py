# %%
import threading



import multiprocessing
from multiprocessing import Process, Queue
from threading import Thread


process1 = Process(target=func, args=(xyz))
process2 = Process(target=func, args=(xyz))

q = Queue()

# getting and putting items in the queue
q.put(1)
q.get()
