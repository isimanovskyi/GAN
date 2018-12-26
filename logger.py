import os as _os
import sys as _sys
import time as _time
import threading

class Logger(object):
    def event(self, msg):
        with open('.\log\event.txt', 'a') as f:
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

