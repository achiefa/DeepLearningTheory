import inspect

import numpy as np


def validate_shape(func):
    """Decorator to validate the shape of the data before adding it to the distribution."""

    def wrapper(self, data):
        if self._shape is not None and data.shape != self._shape:
            raise ValueError(
                f"Data shape {data.shape} does not match expected shape {self._shape}"
            )
        elif self._shape is None:
            self._shape = data.shape
        return func(self, data)

    return wrapper


def validate_size(func):
    """Decorator to validate the size of the distribution before performing operations."""

    def wrapper(self, other):
        if self.size != 0 and self.size != other.size:
            raise ValueError(
                "Number of replicas does not match."
                f"{self.name} is {self.size} and {other.name} is {other.size}"
            )
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


def combine_distributions(list_of_distributions):
    """Combine a list of Distribution objects into a single Distribution."""
    if not list_of_distributions:
        raise ValueError("List of distributions is empty")

    # Get the shape and size from the first distribution
    first = list_of_distributions[0]
    new_shape = (len(list_of_distributions), *first.shape)
    size = first.size
    new_dist = Distribution(name="Combined distribution", shape=new_shape, size=size)

    # Stack the data from each distribution
    combined = np.stack([dist.get_data() for dist in list_of_distributions], axis=1)
    new_dist.set_data(combined)

    return new_dist


class Distribution:
    """A class to represent a distribution of data, with support for
    operations like addition, subtraction, multiplication, and custom
    operations. The distribution can hold multiple replicas of data,
    and each replica can be a multi-dimensional numpy array.

    Attributes:
        name (str): The name of the distribution.
        shape (tuple): The shape of the data of each replica.
        size (int): The number of replicas in the distribution.
    """

    def __init__(self, name, shape=None, size=None):
        """Initialize the Distribution with a name, shape, and size.
        Args:
            name (str): The name of the distribution.
            shape (tuple, optional): The shape of the data for each replica.
                If None, the shape will be determined when data is added.
            size (int, optional): The number of replicas to pre-allocate space for.
                If None, the distribution will grow dynamically.
        """
        if not isinstance(name, str):
            raise TypeError(f"Expected name to be a string, got {type(name)}")

        self.name = name
        self._shape = shape

        self._size = 0
        self._pre_allocated_size = 0 if size is None else size

        if size is not None and shape is not None:
            self._data = [np.zeros(shape=shape) for _ in range(size)]
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

    @validate_size
    def outer(self, other):
        """Compute the outer product of each replica with another distribution."""
        if len(self.shape) > 1 or len(other.shape) > 1:
            raise ValueError("Outer product is only defined for 1D distributions.")
        res = Distribution(
            name=f"{self.name} outer {other.name}",
            shape=(self.shape[0], other.shape[0]),
            size=self.size,
        )
        for rep_self, rep_other in zip(self, other):
            cov = np.outer(rep_self, rep_other)
            res.add(cov)
        return res

    def get_mean(self, axis=0):
        return np.mean(self._data, axis=axis)

    def get_std(self, axis=0):
        return np.std(self._data, axis=axis)

    def get_68_percentile(self, axis=0):
        """Get the 68% percentile of the distribution along a given axis."""
        cl_high = np.nanpercentile(self._data, 84, axis=0)
        cl_low = np.nanpercentile(self._data, 16, axis=0)
        return (cl_low, cl_high)

    def get_data(self):
        """Get the raw data as a numpy multi-dimensional array."""
        return np.array(self._data)

    def set_name(self, name):
        """Set the name of the distribution."""
        if not isinstance(name, str):
            raise TypeError(f"Expected name to be a string, got {type(name)}")
        self.name = name

    def set_data(self, data):
        if self.shape is None or self._pre_allocated_size == 0:
            raise ValueError(
                "Distribution shape and size must be set before adding data."
            )
        if data.shape != (self._pre_allocated_size, *self.shape):
            raise ValueError(
                f"Data shape {data.shape} does not match expected shape {(self._pre_allocated_size, *self.shape)}"
            )
        self._data = data

    def transpose(self):
        """Transpose the data of each replica."""
        res = Distribution(
            f"{self.name} transposed", shape=self.shape[::-1], size=self.size
        )
        for rep in range(self.size):
            res.add(self._data[rep].T)
        return res

    @validate_shape
    @validate_size
    def __add__(self, other):
        """Apply the operator + to each replica"""
        if not isinstance(other, Distribution):
            raise TypeError(f"Expected a Distribution instance, got {type(other)}")

        res = Distribution(
            f"{self.name} + {other.name}", shape=self.shape, size=self.size
        )
        for rep in range(self.size):
            res.add(self._data[rep] + other._data[rep])
        return res

    @validate_shape
    @validate_size
    def __sub__(self, other):
        """Apply the operator - to each replica"""
        if not isinstance(other, Distribution):
            raise TypeError(f"Expected a Distribution instance, got {type(other)}")

        res = Distribution(
            f"{self.name} - {other.name}", shape=self.shape, size=self.size
        )
        for rep in range(self.size):
            res.add(self._data[rep] - other._data[rep])
        return res

    @check_size
    def __mul__(self, other):
        """Multiply the distribution by a scalar"""

        if isinstance(other, (int, float)):
            res = Distribution(
                f"{self.name} * {other}", shape=self.shape, size=self.size
            )
            for rep in range(self.size):
                res.add(self._data[rep] * other)
            return res
        else:
            raise TypeError(f"Expected a scalar, got {type(other)}")

    def __xor__(self, other):
        """Perform outer product with another distribution"""
        if not isinstance(other, Distribution):
            raise TypeError(f"Expected a Distribution instance, got {type(other)}")

        res = self.outer(other)
        return res

    def __matmul__(self, other):
        """Matrix multiplication of the distribution with another distribution"""
        if isinstance(other, Distribution):
            if len(other.shape) < 2:
                shape = (self.shape[0],)
            else:
                shape = (self.shape[0], other.shape[1])
            res = Distribution(
                f"{self.name} @ {other.name}", shape=shape, size=self.size
            )
            for rep in range(self.size):
                res.add(self._data[rep] @ other._data[rep])
            return res

        elif isinstance(other, np.ndarray):
            if len(other.shape) < 2:
                shape = (self.shape[0],)
            else:
                shape = (self.shape[0], other.shape[1])
            res = Distribution(f"{self.name} @ ndarraydd", shape=shape, size=self.size)
            for rep in range(self.size):
                res.add(self._data[rep] @ other)
            return res

        else:
            raise TypeError(f"Expected a Distribution instance, got {type(other)}")

    @check_size
    def __truediv__(self, other):
        """Divide the distribution by a scalar"""
        if isinstance(other, (int, float)):
            res = Distribution(
                f"{self.name} * {other}", shape=self.shape, size=self.size
            )
            for rep in range(self.size):
                res.add(self._data[rep] / other)
            return res
        if isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            res = Distribution(
                f"{self.name} / ndarray", shape=self.shape, size=self.size
            )
            for rep in range(self.size):
                res.add(self._data[rep] / other)
            return res
        else:
            raise TypeError(f"Expected a scalar or array, got {type(other)}")

    @check_size
    def apply_operator(self, b, operator, axis=0, name=None):
        """Apply a custom operator to the distribution along a given axis
        of the data. Operators are expected to be numpy functions that can
        be applied to the data. The operator should be a callable that
        takes a numpy array as input and returns a numpy array."""

        signature = inspect.signature(operator)
        parameters = signature.parameters
        if len(parameters) != 2:
            raise ValueError(
                f"Operator {operator.__name__} must have 2 parameters, got {len(parameters)}"
            )
        if "b" not in parameters:
            raise ValueError(f"Operator {operator.__name__} must have a parameter 'b'")

        name = name if name is not None else f"{self.name} {operator.__name__}"
        res = Distribution(name, shape=self.shape, size=self.size)
        for rep in range(self.size):
            aux = np.apply_along_axis(
                func1d=operator, axis=axis, arr=self._data[rep], b=b
            )
            res.add(aux)
        return res

    def make_diagonal(self):
        """Convert the distribution to a diagonal matrix representation."""
        if self._shape is None or len(self._shape) != 1:
            raise ValueError(
                "Distribution must have a 1D shape to be converted to a diagonal matrix."
            )

        res = Distribution(
            f"{self.name} diagonal",
            shape=(self._shape[0], self._shape[0]),
            size=self.size,
        )
        for rep in range(self.size):
            diag_data = np.diag(self._data[rep])
            res.add(diag_data)
        return res

    def slice(self, index):
        """Slice the distribution data.

        Args:
            index: Can be an int, slice object, tuple of indices/slices, or numpy array
                  Examples:
                  - 5 → data[5]
                  - slice(10, 20) → data[10:20]
                  - (slice(None), 5) → data[:, 5]
                  - (0, slice(10, 20)) → data[0, 10:20]
        """
        sample_sliced = self._data[0][index]
        new_shape = sample_sliced.shape
        res = Distribution(f"{self.name} sliced", shape=new_shape, size=self.size)
        for data in self._data:
            res.add(data[index])
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
        if "_Distribution__data" in state:
            state["_data"] = state.pop("_Distribution__data")
        self.__dict__.update(state)
