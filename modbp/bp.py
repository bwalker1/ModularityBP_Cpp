# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_bp')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_bp')
    _bp = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_bp', [dirname(__file__)])
        except ImportError:
            import _bp
            return _bp
        try:
            _mod = imp.load_module('_bp', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _bp = swig_import_helper()
    del swig_import_helper
else:
    import _bp
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0


def print_array(arr, n):
    return _bp.print_array(arr, n)
print_array = _bp.print_array
class BP_Modularity(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BP_Modularity, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BP_Modularity, name)
    __repr__ = _swig_repr

    def __init__(self, layer_membership, intra_edgelist, inter_edgelist, _n, _nt, q, beta, omega=1.0, resgamma=1.0, verbose=False, transform=False):
        this = _bp.new_BP_Modularity(layer_membership, intra_edgelist, inter_edgelist, _n, _nt, q, beta, omega, resgamma, verbose, transform)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def run(self, maxIters=100):
        return _bp.BP_Modularity_run(self, maxIters)

    def step(self):
        return _bp.BP_Modularity_step(self)

    def compute_marginals(self):
        return _bp.BP_Modularity_compute_marginals(self)

    def compute_bethe_free_energy(self):
        return _bp.BP_Modularity_compute_bethe_free_energy(self)

    def compute_factorized_free_energy(self):
        return _bp.BP_Modularity_compute_factorized_free_energy(self)

    def return_marginals(self):
        return _bp.BP_Modularity_return_marginals(self)

    def getBeta(self):
        return _bp.BP_Modularity_getBeta(self)

    def setBeta(self, arg2, reset=True):
        return _bp.BP_Modularity_setBeta(self, arg2, reset)

    def getResgamma(self):
        return _bp.BP_Modularity_getResgamma(self)

    def setResgamma(self, arg2, reset=True):
        return _bp.BP_Modularity_setResgamma(self, arg2, reset)

    def getOmega(self):
        return _bp.BP_Modularity_getOmega(self)

    def setOmega(self, arg2, reset=True):
        return _bp.BP_Modularity_setOmega(self, arg2, reset)

    def getq(self):
        return _bp.BP_Modularity_getq(self)

    def setq(self, new_q):
        return _bp.BP_Modularity_setq(self, new_q)

    def set_compute_bfe(self, b):
        return _bp.BP_Modularity_set_compute_bfe(self, b)

    def getVerbose(self):
        return _bp.BP_Modularity_getVerbose(self)

    def setVerbose(self, arg2):
        return _bp.BP_Modularity_setVerbose(self, arg2)

    def compute_bstar(self):
        return _bp.BP_Modularity_compute_bstar(self)
    __swig_destroy__ = _bp.delete_BP_Modularity
    __del__ = lambda self: None
BP_Modularity_swigregister = _bp.BP_Modularity_swigregister
BP_Modularity_swigregister(BP_Modularity)

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _bp.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return _bp.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _bp.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _bp.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _bp.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _bp.SwigPyIterator_equal(self, x)

    def copy(self):
        return _bp.SwigPyIterator_copy(self)

    def next(self):
        return _bp.SwigPyIterator_next(self)

    def __next__(self):
        return _bp.SwigPyIterator___next__(self)

    def previous(self):
        return _bp.SwigPyIterator_previous(self)

    def advance(self, n):
        return _bp.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _bp.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _bp.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _bp.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _bp.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _bp.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _bp.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _bp.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class PairVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PairVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PairVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _bp.PairVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _bp.PairVector___nonzero__(self)

    def __bool__(self):
        return _bp.PairVector___bool__(self)

    def __len__(self):
        return _bp.PairVector___len__(self)

    def __getslice__(self, i, j):
        return _bp.PairVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _bp.PairVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _bp.PairVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _bp.PairVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _bp.PairVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _bp.PairVector___setitem__(self, *args)

    def pop(self):
        return _bp.PairVector_pop(self)

    def append(self, x):
        return _bp.PairVector_append(self, x)

    def empty(self):
        return _bp.PairVector_empty(self)

    def size(self):
        return _bp.PairVector_size(self)

    def swap(self, v):
        return _bp.PairVector_swap(self, v)

    def begin(self):
        return _bp.PairVector_begin(self)

    def end(self):
        return _bp.PairVector_end(self)

    def rbegin(self):
        return _bp.PairVector_rbegin(self)

    def rend(self):
        return _bp.PairVector_rend(self)

    def clear(self):
        return _bp.PairVector_clear(self)

    def get_allocator(self):
        return _bp.PairVector_get_allocator(self)

    def pop_back(self):
        return _bp.PairVector_pop_back(self)

    def erase(self, *args):
        return _bp.PairVector_erase(self, *args)

    def __init__(self, *args):
        this = _bp.new_PairVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _bp.PairVector_push_back(self, x)

    def front(self):
        return _bp.PairVector_front(self)

    def back(self):
        return _bp.PairVector_back(self)

    def assign(self, n, x):
        return _bp.PairVector_assign(self, n, x)

    def resize(self, *args):
        return _bp.PairVector_resize(self, *args)

    def insert(self, *args):
        return _bp.PairVector_insert(self, *args)

    def reserve(self, n):
        return _bp.PairVector_reserve(self, n)

    def capacity(self):
        return _bp.PairVector_capacity(self)
    __swig_destroy__ = _bp.delete_PairVector
    __del__ = lambda self: None
PairVector_swigregister = _bp.PairVector_swigregister
PairVector_swigregister(PairVector)

class IntArray(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, IntArray, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, IntArray, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _bp.IntArray_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _bp.IntArray___nonzero__(self)

    def __bool__(self):
        return _bp.IntArray___bool__(self)

    def __len__(self):
        return _bp.IntArray___len__(self)

    def __getslice__(self, i, j):
        return _bp.IntArray___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _bp.IntArray___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _bp.IntArray___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _bp.IntArray___delitem__(self, *args)

    def __getitem__(self, *args):
        return _bp.IntArray___getitem__(self, *args)

    def __setitem__(self, *args):
        return _bp.IntArray___setitem__(self, *args)

    def pop(self):
        return _bp.IntArray_pop(self)

    def append(self, x):
        return _bp.IntArray_append(self, x)

    def empty(self):
        return _bp.IntArray_empty(self)

    def size(self):
        return _bp.IntArray_size(self)

    def swap(self, v):
        return _bp.IntArray_swap(self, v)

    def begin(self):
        return _bp.IntArray_begin(self)

    def end(self):
        return _bp.IntArray_end(self)

    def rbegin(self):
        return _bp.IntArray_rbegin(self)

    def rend(self):
        return _bp.IntArray_rend(self)

    def clear(self):
        return _bp.IntArray_clear(self)

    def get_allocator(self):
        return _bp.IntArray_get_allocator(self)

    def pop_back(self):
        return _bp.IntArray_pop_back(self)

    def erase(self, *args):
        return _bp.IntArray_erase(self, *args)

    def __init__(self, *args):
        this = _bp.new_IntArray(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _bp.IntArray_push_back(self, x)

    def front(self):
        return _bp.IntArray_front(self)

    def back(self):
        return _bp.IntArray_back(self)

    def assign(self, n, x):
        return _bp.IntArray_assign(self, n, x)

    def resize(self, *args):
        return _bp.IntArray_resize(self, *args)

    def insert(self, *args):
        return _bp.IntArray_insert(self, *args)

    def reserve(self, n):
        return _bp.IntArray_reserve(self, n)

    def capacity(self):
        return _bp.IntArray_capacity(self)
    __swig_destroy__ = _bp.delete_IntArray
    __del__ = lambda self: None
IntArray_swigregister = _bp.IntArray_swigregister
IntArray_swigregister(IntArray)

class Array(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Array, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Array, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _bp.Array_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _bp.Array___nonzero__(self)

    def __bool__(self):
        return _bp.Array___bool__(self)

    def __len__(self):
        return _bp.Array___len__(self)

    def __getslice__(self, i, j):
        return _bp.Array___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _bp.Array___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _bp.Array___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _bp.Array___delitem__(self, *args)

    def __getitem__(self, *args):
        return _bp.Array___getitem__(self, *args)

    def __setitem__(self, *args):
        return _bp.Array___setitem__(self, *args)

    def pop(self):
        return _bp.Array_pop(self)

    def append(self, x):
        return _bp.Array_append(self, x)

    def empty(self):
        return _bp.Array_empty(self)

    def size(self):
        return _bp.Array_size(self)

    def swap(self, v):
        return _bp.Array_swap(self, v)

    def begin(self):
        return _bp.Array_begin(self)

    def end(self):
        return _bp.Array_end(self)

    def rbegin(self):
        return _bp.Array_rbegin(self)

    def rend(self):
        return _bp.Array_rend(self)

    def clear(self):
        return _bp.Array_clear(self)

    def get_allocator(self):
        return _bp.Array_get_allocator(self)

    def pop_back(self):
        return _bp.Array_pop_back(self)

    def erase(self, *args):
        return _bp.Array_erase(self, *args)

    def __init__(self, *args):
        this = _bp.new_Array(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _bp.Array_push_back(self, x)

    def front(self):
        return _bp.Array_front(self)

    def back(self):
        return _bp.Array_back(self)

    def assign(self, n, x):
        return _bp.Array_assign(self, n, x)

    def resize(self, *args):
        return _bp.Array_resize(self, *args)

    def insert(self, *args):
        return _bp.Array_insert(self, *args)

    def reserve(self, n):
        return _bp.Array_reserve(self, n)

    def capacity(self):
        return _bp.Array_capacity(self)
    __swig_destroy__ = _bp.delete_Array
    __del__ = lambda self: None
Array_swigregister = _bp.Array_swigregister
Array_swigregister(Array)

# This file is compatible with both classic and new-style classes.


