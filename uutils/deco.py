import time
import functools
import sys


def stdout(file):
    def deco(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):

            sys.stdout = open(file, "w")
            result = func(*args, **kwargs)
            sys.stdout.close()
            return result

        return wrap

    return deco


def repeat(func=None, *, n=2):
    def deco(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):

            for _ in range(n):
                result = func(*args, **kwargs)

            return result

        return wrap

    return deco if func is None else deco(func)


def timer(func=None):
    @functools.wraps(func)
    def wrap(*args, **kwargs):

        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f"Done (t={round(toc-tic,2)})s")
        return result

    return wrap


class DecoratorClass:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("decorated class")
        return self.func(*args, **kwargs)


def singleton(cls):
    """Make a class a Singleton class (only one instance)"""

    @functools.wraps(cls)
    def wrap(*args, **kwargs):
        if not wrap.instance:
            wrap.instance = cls(*args, **kwargs)
        return wrap.instance

    wrap.instance = None
    return wrap


@singleton
class TheOne:
    "example"
    pass


def sleep(func=None, *, t=1):
    """Sleep given amount of seconds before calling the function"""

    def deco(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):

            time.sleep(t)
            return func(*args, **kwargs)

        return wrap

    return deco if func is None else deco(func)


def cache(func):
    """Keep a cache of previous function calls"""
    '''use @functools.lru_cache'''

    @functools.wraps(func)
    def wrap(*args, **kwargs):

        try:
            return wrap.cache[args, kwargs]
        except:
            wrap.cache[args, kwargs] = func(*args, **kwargs)
            return wrap.cache[args, kwargs]

    wrap.cache = dict()
    return wrap

PLUGINS = dict()
def register(func):
    """Register a function as a plug-in"""

    PLUGINS[func.__name__] = func
    return func
