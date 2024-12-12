from enum import Enum, EnumMeta

# Allows the lookup for enum values by name to become case-insensitive
# (under the assumption that the enum names will be all uppercase)
# https://stackoverflow.com/a/51869269
class CaseInsensitveEnumMeta(EnumMeta):
    def __getitem__(self, item):
        if isinstance(item, str):
            item = item.upper()

        return super().__getitem__(item)

class Coordinate2D(Enum, metaclass=CaseInsensitveEnumMeta):
    X = 0
    Y = 1

class Coordinate3D(Enum, metaclass=CaseInsensitveEnumMeta):
    X = 0
    Y = 1
    Z = 2
