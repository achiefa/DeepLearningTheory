import numpy as np
import inspect

def validate_shape(func):
  def wrapper(self, data):
    if self._shape is not None and data.shape != self._shape:
      raise ValueError(f"Data shape {data.shape} does not match expected shape {self._shape}")
    elif self._shape is None:
      self._shape = data.shape
    return func(self, data)
  return wrapper

def validate_size(func):
  def wrapper(self, other):
    if self.size != 0 and self.size != other.size:
      raise ValueError("Number of replicas does not match."
                       f"{self.name} is {self.size} and {other.name} is {other.size}")
    return func(self, other)
  return wrapper

def check_size(func):
  def wrapper(self, *args, **kwargs):
    if self.size == 0:
      raise ValueError("Distribution has no replicas")
    return func(self, *args, **kwargs)
  return wrapper

def numpify(func):
  def wrapper(self, *args, **kwargs):
    if len(args) == 1 and isinstance(args[0], np.ndarray):
      return func(self, *args, **kwargs)
    else:
      return func(self, np.array(args), **kwargs)
  return wrapper
    

class Distribution():
  def __init__(self, name, shape=None, size=None):
    self.name = name
    self._shape = shape
    
    self._size = 0
    self._pre_allocated_size = 0 if size is None else size

    if size is not None and shape is not None:
      self._data = [np.zeros(*shape) for _ in range(size)]
    else:
      self._data = []
  
  @property
  def size(self):
    return self._size
  
  @property
  def shape(self):
    return self._shape

  @validate_shape
  def add(self, data):
    if self._size < self._pre_allocated_size:
      self._data[self._size] = data
      self._size += 1
    else:
      self._data.append(data)
      self._size += 1

  def get_mean(self, axis=0):
    return np.mean(self._data, axis=axis)
  
  def get_std(self, axis=0):
    return np.std(self._data, axis=axis)
  
  def get_data(self):
    """Get the raw data as a numpy multi-dimensional array."""
    return np.array(self._data)
  
  @validate_shape
  @validate_size
  def __add__(self, other):
    """Apply the operator + to each replica"""
    if not isinstance(other, Distribution):
      raise TypeError(f"Expected a Distribution instance, got {type(other)}")
    
    res = Distribution(f"{self.name} + {other.name}", shape=self.shape, size=self.size)
    for rep in range(self.size):
      res.add(self._data[rep] + other._data[rep])
    return res

  @validate_shape
  @validate_size
  def __sub__(self, other):
    """Apply the operator - to each replica"""
    if not isinstance(other, Distribution):
      raise TypeError(f"Expected a Distribution instance, got {type(other)}")
    
    res = Distribution(f"{self.name} - {other.name}", shape=self.shape, size=self.size)
    for rep in range(self.size):
      res.add(self._data[rep] - other._data[rep])
    return res
  
  @check_size
  def __mul__(self, other):
    """Multiply the distribution by a scalar"""
    
    if isinstance(other, (int, float)):
      res = Distribution(f"{self.name} * {other}", shape=self.shape, size=self.size)
      for rep in range(self.size):
        res.add(self._data[rep] * other)
      return res
    else:
      raise TypeError(f"Expected a scalar, got {type(other)}")
    
  def __matmul__(self, other):
    if not isinstance(other, Distribution):
      raise TypeError(f"Expected a Distribution instance, got {type(other)}")
    
    shape = np.array([self.shape[0], other.shape[1]])
    res = Distribution(f"{self.name} @ {other.name}", shape=shape, size=self.size)
    for rep in range(self.size):
      res.add(self._data[rep] @ other._data[rep])
    return res
    
  @check_size
  def __truediv__(self, other):
    """Multiply the distribution by a scalar"""
    if isinstance(other, (int, float)):
      res = Distribution(f"{self.name} * {other}", shape=self.shape, size=self.size)
      for rep in range(self.size):
        res.add(self._data[rep] / other)
      return res
    else:
      raise TypeError(f"Expected a scalar, got {type(other)}")
  
  @check_size
  def apply_operator(self, b, operator, axis=0, name=None):
    """Apply a custom operator to the distribution along a given axis
    of the data. Operators are expected to be numpy functions that can
    be applied to the data. The operator should be a callable that
    takes a numpy array as input and returns a numpy array."""

    signature = inspect.signature(operator)
    parameters = signature.parameters
    if len(parameters) != 2:
      raise ValueError(f"Operator {operator.__name__} must have 2 parameters, got {len(parameters)}")
    if 'b' not in parameters:
      raise ValueError(f"Operator {operator.__name__} must have a parameter 'b'")
    
    name = name if name is not None else f"{self.name} {operator.__name__}"
    res = Distribution(name, shape=self.shape, size=self.size)
    for rep in range(self.size):
      aux = np.apply_along_axis(func1d=operator, axis=axis, arr=self._data[rep], b=b)
      res.add(aux)
    return res
    
  def __str__(self):
    return f"{self.name}:\n{self._data}"
  
  def __repr__(self):
    return self.__str__()
  
  def __len__(self):
    return self._size

  def __getitem__(self, key):
    return self._data[key]
  
  def __setstate__(self, state):
    """Handle deserialization and patch old serialized objects."""
    # If the old `_data` attribute exists, rename it to `__data`
    if '_Distribution__data' in state:
      state['_data'] = state.pop('_Distribution__data')
    self.__dict__.update(state)