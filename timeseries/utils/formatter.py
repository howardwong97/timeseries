import numpy as np

__all__ = ["format_float_fixed"]


def format_float_fixed(x: float, max_digits: int = 10, decimal: int = 4) -> str:
    """Formats a floating point number so that if it can be well expressed
    in using a string with digits len, then it is converted simply, otherwise
    it is expressed in scientific notation"""
    # basic_format = '{:0.' + str(digits) + 'g}'
    if x == 0:
        return ("{:0." + str(decimal) + "f}").format(0.0)
    scale = np.log10(np.abs(x))
    scale = np.sign(scale) * np.ceil(np.abs(scale))
    if scale > (max_digits - 2 - decimal) or scale < -(decimal - 2):
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "e}").format(x)
    else:
        formatted = ("{0:" + str(max_digits) + "." + str(decimal) + "f}").format(x)
    return formatted
