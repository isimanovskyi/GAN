import os as _os
import sys as _sys
import time as _time
import threading

class Logger(object):
    def __init__(self):
        self.in_progress = False
        self.max_progress = 0

    def event(self, msg):
        with open('./checkpoint/event.txt', 'a') as f:
            f.write(msg + '\n')

    def log(self, level, msg, *args, **kwargs):
        print (msg)

    def debug(self, msg, *args, **kwargs):
        print (msg)

    def info(self, msg, *args, **kwargs):
        print (msg)

    def error(self, msg, *args, **kwargs):
        print (msg)

    def fatal(self, msg, *args, **kwargs):
        print (msg)

    def warning(self, msg, *args, **kwargs):
        print (msg)

    def start_progress(self, max_progress):
        if self.in_progress:
            raise RuntimeError('Already in progress')
        self.in_progress = True
        self.max_progress = max_progress

    def end_progress(self):
        if not self.in_progress:
            raise RuntimeError('Not in progress')
        self.in_progress = False
        print ("")


    def progress(self, i, msg):
        if not self.in_progress:
            raise RuntimeError('Not in progress')

        if i >= self.max_progress:
            raise ValueError('progress counter too big')

        _sys.stdout.write('\r' + '[%02d/%02d]-' % (i+1, self.max_progress) + msg)
        _sys.stdout.flush()

_logger = None
_logger_lock = threading.Lock()

def _get_logger():
    global _logger

    if _logger is not None:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger:
            return _logger

        _logger = Logger()
        return _logger

    finally:
        _logger_lock.release()

def event(msg):
    _get_logger().event(msg)
    

def log(level, msg, *args, **kwargs):
    _get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    _get_logger().debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    _get_logger().info(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    _get_logger().error("ERROR: %s" % msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    _get_logger().fatal("FATAL: %s" % msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    _get_logger().warning("WARNING: %s" % msg, *args, **kwargs)

def start_progress(max_progress):
    _get_logger().start_progress(max_progress)

def end_progress():
    _get_logger().end_progress()

def progress(i, msg):
    _get_logger().progress(i,msg)

