import numpy as np
import pytest

from yadlt.distribution import Distribution, combine_distributions


class TestDistribution:
    """Test suite for the Distribution class."""

    def test_init_basic(self):
        """Test basic initialization."""
        dist = Distribution("test")
        assert dist.name == "test"
        assert dist.size == 0
        assert dist.shape is None

    def test_init_with_shape_and_size(self):
        """Test initialization with shape and size."""
        dist = Distribution("test", shape=(5,), size=10)
        assert dist.name == "test"
        assert dist.size == 0
        assert dist.shape == (5,)
        assert dist._pre_allocated_size == 10

    def test_init_invalid_name(self):
        """Test initialization with invalid name type."""
        with pytest.raises(TypeError):
            Distribution(123)

    def test_add_data(self):
        """Test adding data to distribution."""
        dist = Distribution("test")
        data = np.array([1, 2, 3])
        dist.add(data)

        assert dist.size == 1
        assert dist.shape == (3,)
        np.testing.assert_array_equal(dist[0], data)

    def test_add_data_shape_mismatch(self):
        """Test adding data with mismatched shape."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        with pytest.raises(ValueError):
            dist.add(np.array([1, 2]))

    def test_add_data_preallocated(self):
        """Test adding data to pre-allocated distribution."""
        dist = Distribution("test", shape=(3,), size=2)
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        dist.add(data1)
        dist.add(data2)

        assert dist.size == 2
        np.testing.assert_array_equal(dist[0], data1)
        np.testing.assert_array_equal(dist[1], data2)

    def test_get_mean(self):
        """Test computing mean."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))
        dist.add(np.array([4, 5, 6]))

        mean = dist.get_mean()
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_equal(mean, expected)

    def test_get_std(self):
        """Test computing standard deviation."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))
        dist.add(np.array([4, 5, 6]))

        std = dist.get_std()
        expected = np.array([1.5, 1.5, 1.5])
        np.testing.assert_array_equal(std, expected)

    def test_get_68_percentile(self):
        """Test computing 68% percentile."""
        dist = Distribution("test")
        for i in range(100):
            dist.add(np.array([i]))

        low, high = dist.get_68_percentile()
        assert low[0] == pytest.approx(15.84, rel=1e-2)
        assert high[0] == pytest.approx(83.16, rel=1e-2)

    def test_get_data(self):
        """Test getting raw data."""
        dist = Distribution("test")
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        dist.add(data1)
        dist.add(data2)

        raw_data = dist.get_data()
        expected = np.array([data1, data2])
        np.testing.assert_array_equal(raw_data, expected)

    def test_set_name(self):
        """Test setting distribution name."""
        dist = Distribution("test")
        dist.set_name("new_name")
        assert dist.name == "new_name"

    def test_set_name_invalid_type(self):
        """Test setting name with invalid type."""
        dist = Distribution("test")
        with pytest.raises(TypeError):
            dist.set_name(123)

    def test_set_data(self):
        """Test setting data directly."""
        dist = Distribution("test", shape=(2,), size=2)
        data = np.array([[1, 2], [3, 4]])
        dist.set_data(data)

        np.testing.assert_array_equal(dist.get_data(), data)

    def test_set_data_invalid_shape(self):
        """Test setting data with invalid shape."""
        dist = Distribution("test", shape=(2,), size=2)
        data = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError):
            dist.set_data(data)

    def test_set_data_no_preallocation(self):
        """Test setting data without pre-allocation."""
        dist = Distribution("test")
        data = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError):
            dist.set_data(data)

    def test_transpose(self):
        """Test transposing distribution data."""
        dist = Distribution("test")
        dist.add(np.array([[1, 2], [3, 4]]))
        dist.add(np.array([[5, 6], [7, 8]]))

        transposed = dist.transpose()

        assert transposed.shape == (2, 2)
        expected1 = np.array([[1, 3], [2, 4]])
        expected2 = np.array([[5, 7], [6, 8]])
        np.testing.assert_array_equal(transposed[0], expected1)
        np.testing.assert_array_equal(transposed[1], expected2)

    def test_addition(self):
        """Test distribution addition."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")

        dist1.add(np.array([1, 2, 3]))
        dist1.add(np.array([4, 5, 6]))

        dist2.add(np.array([10, 20, 30]))
        dist2.add(np.array([40, 50, 60]))

        result = dist1 + dist2

        assert result.name == "dist1 + dist2"
        expected1 = np.array([11, 22, 33])
        expected2 = np.array([44, 55, 66])
        np.testing.assert_array_equal(result[0], expected1)
        np.testing.assert_array_equal(result[1], expected2)

    def test_addition_size_mismatch(self):
        """Test addition with size mismatch."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")

        dist1.add(np.array([1, 2, 3]))
        dist2.add(np.array([1, 2, 3]))
        dist2.add(np.array([4, 5, 6]))

        with pytest.raises(ValueError):
            dist1 + dist2

    def test_subtraction(self):
        """Test distribution subtraction."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")

        dist1.add(np.array([10, 20, 30]))
        dist2.add(np.array([1, 2, 3]))

        result = dist1 - dist2

        assert result.name == "dist1 - dist2"
        expected = np.array([9, 18, 27])
        np.testing.assert_array_equal(result[0], expected)

    def test_scalar_multiplication(self):
        """Test multiplication by scalar."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        result = dist * 5

        assert result.name == "test * 5"
        expected = np.array([5, 10, 15])
        np.testing.assert_array_equal(result[0], expected)

    def test_scalar_multiplication_invalid_type(self):
        """Test multiplication with invalid type."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        with pytest.raises(TypeError):
            dist * "invalid"

    def test_scalar_division(self):
        """Test division by scalar."""
        dist = Distribution("test")
        dist.add(np.array([10, 20, 30]))

        result = dist / 2

        expected = np.array([5, 10, 15])
        np.testing.assert_array_equal(result[0], expected)

    def test_array_division(self):
        """Test division by array."""
        dist = Distribution("test")
        dist.add(np.array([10, 20, 30]))

        divisor = np.array([2, 4, 5])
        result = dist / divisor

        expected = np.array([5, 5, 6])
        np.testing.assert_array_equal(result[0], expected)

    def test_array_division_shape_mismatch(self):
        """Test division by array with shape mismatch."""
        dist = Distribution("test")
        dist.add(np.array([10, 20, 30]))

        divisor = np.array([2, 4])

        with pytest.raises(ValueError):
            dist / divisor

    def test_outer_product(self):
        """Test outer product between distributions."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")

        dist1.add(np.array([1, 2]))
        dist2.add(np.array([3, 4]))

        result = dist1.outer(dist2)

        assert result.name == "dist1 outer dist2"
        assert result.shape == (2, 2)
        expected = np.array([[3, 4], [6, 8]])
        np.testing.assert_array_equal(result[0], expected)

    def test_outer_product_operator(self):
        """Test outer product using ^ operator."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")

        dist1.add(np.array([1, 2]))
        dist2.add(np.array([3, 4]))

        result = dist1 ^ dist2

        expected = np.array([[3, 4], [6, 8]])
        np.testing.assert_array_equal(result[0], expected)

    def test_outer_product_invalid_dimension(self):
        """Test outer product with invalid dimensions."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")

        dist1.add(np.array([[1, 2], [3, 4]]))
        dist2.add(np.array([1, 2]))

        with pytest.raises(ValueError):
            dist1.outer(dist2)

    def test_matrix_multiplication_distributions(self):
        """Test matrix multiplication between distributions."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")

        dist1.add(np.array([[1, 2], [3, 4]]))
        dist2.add(np.array([[5, 6], [7, 8]]))

        result = dist1 @ dist2

        assert result.name == "dist1 @ dist2"
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result[0], expected)

    def test_matrix_multiplication_array(self):
        """Test matrix multiplication with numpy array."""
        dist = Distribution("test")
        dist.add(np.array([[1, 2], [3, 4]]))

        array = np.array([[5, 6], [7, 8]])
        result = dist @ array

        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result[0], expected)

    def test_matrix_multiplication_vector(self):
        """Test matrix multiplication with vector."""
        dist = Distribution("test")
        dist.add(np.array([[1, 2], [3, 4]]))

        vector = np.array([5, 6])
        result = dist @ vector

        assert result.shape == (2,)
        expected = np.array([17, 39])
        np.testing.assert_array_equal(result[0], expected)

    def test_apply_operator(self):
        """Test applying custom operator."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))
        dist.add(np.array([4, 5, 6]))

        def custom_op(a, b):
            return a + b

        result = dist.apply_operator(10, custom_op)

        expected1 = np.array([11, 12, 13])
        expected2 = np.array([14, 15, 16])
        np.testing.assert_array_equal(result[0], expected1)
        np.testing.assert_array_equal(result[1], expected2)

    def test_apply_operator_invalid_signature(self):
        """Test applying operator with invalid signature."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        def invalid_op(a):
            return a + 1

        with pytest.raises(ValueError):
            dist.apply_operator(10, invalid_op)

    def test_apply_operator_missing_b_parameter(self):
        """Test applying operator missing 'b' parameter."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        def invalid_op(a, c):
            return a + c

        with pytest.raises(ValueError):
            dist.apply_operator(10, invalid_op)

    def test_make_diagonal(self):
        """Test converting 1D distribution to diagonal matrix."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        result = dist.make_diagonal()

        assert result.shape == (3, 3)
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        np.testing.assert_array_equal(result[0], expected)

    def test_make_diagonal_invalid_shape(self):
        """Test make_diagonal with invalid shape."""
        dist = Distribution("test")
        dist.add(np.array([[1, 2], [3, 4]]))

        with pytest.raises(ValueError):
            dist.make_diagonal()

    def test_slice(self):
        """Test slicing distribution data."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3, 4, 5]))
        dist.add(np.array([6, 7, 8, 9, 10]))

        result = dist.slice((slice(1, 4),))

        assert result.shape == (3,)
        expected1 = np.array([2, 3, 4])
        expected2 = np.array([7, 8, 9])
        np.testing.assert_array_equal(result[0], expected1)
        np.testing.assert_array_equal(result[1], expected2)

    def test_slice_invalid_index(self):
        """Test slicing with invalid index."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        with pytest.raises(IndexError):
            dist.slice((5,))

    def test_bootstrap(self):
        """Test bootstrap resampling."""
        dist = Distribution("test")
        for i in range(10):
            dist.add(np.array([i]))

        bootstrap_result = dist.bootstrap(size_bootstrap=100, seed=42)

        assert bootstrap_result.size == 100
        assert bootstrap_result.shape == (1,)
        assert bootstrap_result.name == "test bootstrap"

    def test_empty_distribution_operations(self):
        """Test operations on empty distributions."""
        dist = Distribution("test")

        with pytest.raises(ValueError):
            dist.get_mean()

        with pytest.raises(ValueError):
            dist * 5

        with pytest.raises(ValueError):
            dist / 2

    def test_string_representation(self):
        """Test string representation."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        str_repr = str(dist)
        assert "test" in str_repr

        repr_str = repr(dist)
        assert "test" in repr_str

    def test_len(self):
        """Test length of distribution."""
        dist = Distribution("test")
        assert len(dist) == 0

        dist.add(np.array([1, 2, 3]))
        assert len(dist) == 1

        dist.add(np.array([4, 5, 6]))
        assert len(dist) == 2

    def test_getitem(self):
        """Test indexing into distribution."""
        dist = Distribution("test")
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        dist.add(data1)
        dist.add(data2)

        np.testing.assert_array_equal(dist[0], data1)
        np.testing.assert_array_equal(dist[1], data2)

    def test_setstate(self):
        """Test state restoration for backward compatibility."""
        dist = Distribution("test")

        # Simulate old state format
        old_state = {
            "name": "test",
            "_shape": (3,),
            "_size": 1,
            "_pre_allocated_size": 0,
            "_Distribution__data": [np.array([1, 2, 3])],
        }

        dist.__setstate__(old_state)

        assert dist.name == "test"
        assert dist.shape == (3,)
        assert dist.size == 1
        np.testing.assert_array_equal(dist[0], np.array([1, 2, 3]))


class TestCombineDistributions:
    """Test suite for the combine_distributions function."""

    def test_combine_empty_list(self):
        """Test combining empty list of distributions."""
        with pytest.raises(ValueError):
            combine_distributions([])

    def test_combine_single_distribution(self):
        """Test combining a single distribution."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        result = combine_distributions([dist])

        assert result.shape == (1, 3)
        assert result.size == 1

        np.testing.assert_array_equal(result[0], np.array([[1, 2, 3]]))

    def test_combine_multiple_distributions(self):
        """Test combining multiple distributions."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")

        dist1.add(np.array([1, 2]))
        dist1.add(np.array([3, 4]))

        dist2.add(np.array([5, 6]))
        dist2.add(np.array([7, 8]))

        result = combine_distributions([dist1, dist2])

        assert result.shape == (2, 2)
        assert result.size == 2
        expected1 = np.array([[1, 2], [5, 6]])
        expected2 = np.array([[3, 4], [7, 8]])
        np.testing.assert_array_equal(result[0], expected1)
        np.testing.assert_array_equal(result[1], expected2)


class TestDistributionDecorators:
    """Test suite for testing decorator behavior."""

    def test_validate_shape_decorator(self):
        """Test that shape validation works correctly."""
        dist = Distribution("test")
        dist.add(np.array([1, 2, 3]))

        # This should work - same shape
        dist.add(np.array([4, 5, 6]))

        # This should fail - different shape
        with pytest.raises(ValueError):
            dist.add(np.array([1, 2]))

    def test_validate_size_decorator(self):
        """Test that size validation works correctly."""
        dist1 = Distribution("dist1")
        dist2 = Distribution("dist2")
        dist3 = Distribution("dist3")

        dist1.add(np.array([1, 2, 3]))
        dist1.add(np.array([4, 5, 6]))

        dist2.add(np.array([10, 20, 30]))
        dist2.add(np.array([40, 50, 60]))

        # This should work - same size
        result = dist1 + dist2
        assert result.size == 2

        # This should fail - different size
        dist3.add(np.array([100, 200, 300]))

        with pytest.raises(ValueError):
            dist1 + dist3

    def test_check_size_decorator(self):
        """Test that size checking works correctly."""
        dist = Distribution("test")

        # These should fail on empty distribution
        with pytest.raises(ValueError):
            dist * 5

        with pytest.raises(ValueError):
            dist / 2

        with pytest.raises(ValueError):
            dist.apply_operator(10, lambda a, b: a + b)

        # After adding data, operations should work
        dist.add(np.array([1, 2, 3]))
        result = dist * 5
        assert result.size == 1
