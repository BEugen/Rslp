from multiprocessing import Process, Queue
import time


def f(q):
    time.sleep(5)
    q.put([42, None, 'hello'])


if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    while p.is_alive():
        print("Wait")
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()
