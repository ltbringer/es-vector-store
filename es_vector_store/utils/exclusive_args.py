# Write a function that expects n arguments.
# The function should return True if and only if exactly one of the arguments is not None.
# Otherwise, it should return False.
from enum import Enum


class MutuallyExclusiveStatusCode(Enum):
    OK = 0
    ALL_NONE = 1
    MULTIPLE_SET = 2


def is_mutually_exclusive(*args):
    all_nones = all(arg is None for arg in args)
    multiple_set = sum(1 for arg in args if arg is not None) > 1

    if all_nones:
        return MutuallyExclusiveStatusCode.ALL_NONE
    if multiple_set:
        return MutuallyExclusiveStatusCode.MULTIPLE_SET
    return MutuallyExclusiveStatusCode.OK
