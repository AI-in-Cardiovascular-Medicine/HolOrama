import numpy as np

from input_output.output.contours import _to_serializable


class TestToSerializable:
    def test_numpy_float_scalar_to_python_float(self):
        result = _to_serializable(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_int_scalar_to_python_int(self):
        result = _to_serializable(np.int32(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_array_to_list(self):
        result = _to_serializable(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_2d_array_to_nested_list(self):
        result = _to_serializable(np.array([[1, 2], [3, 4]]))
        assert result == [[1, 2], [3, 4]]

    def test_unknown_type_falls_back_to_str(self):
        class Custom:
            def __str__(self):
                return 'custom_value'

        assert _to_serializable(Custom()) == 'custom_value'
