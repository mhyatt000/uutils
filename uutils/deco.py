import functools
import sys

import transformers


def stdout(file):
    def deco(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):

            sys.stdout = open(file, "w")
            result = func()
            sys.stdout.close()
            return result

        return wrap

    return deco


def repeat(n):
    def deco(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):

            for _ in range(n):
                result = func(*args, **kwargs)

            return result

        return wrap

    r0eturn deco

def timer():
    def deco(func)
        @functools.wraps(func)
        def weap(*args, **kwargs):

            tic = time.perf_timer()
            result = func(*args, **kwargs)
            toc = time.per_timer()

            print(f'Done (t={round(toc-tic,2)}s)')
            return result

        return wrap
    return deco
