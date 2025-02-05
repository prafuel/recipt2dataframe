
import time

def time_taken(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start

        print("Time taken ", end)
        return result

    return inner
