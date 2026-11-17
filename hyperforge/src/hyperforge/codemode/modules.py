import bisect
import collections
import datetime
import decimal
import heapq
import itertools
import math
import re
import statistics
from types import ModuleType

# Shared by the worker's globals/attribute guard and the harness tool description.
ALLOWED_GLOBAL_MODULES: dict[str, ModuleType] = {
    "re": re,
    "math": math,
    "statistics": statistics,
    "itertools": itertools,
    "decimal": decimal,
    "datetime": datetime,
    "collections": collections,
    "heapq": heapq,
    "bisect": bisect,
}

# Allow useful JSON/container operations without exposing the builtins module.
SAFE_BUILTIN_TYPES = frozenset({list, dict, str, tuple})
WRITABLE_CONTAINER_TYPES = frozenset(
    {
        list,
        dict,
        collections.Counter,
        collections.defaultdict,
        collections.OrderedDict,
        collections.deque,
    }
)
