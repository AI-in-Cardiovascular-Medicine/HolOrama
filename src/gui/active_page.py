from enum import Enum

from domain.io_types import MetaDataIntravascular, MetaDataCCTA


class ActivePage(Enum):
    INTRAVASCULAR = 0
    CCTA = 1

    @classmethod
    def from_index(cls, index: int) -> 'ActivePage':
        for page in cls:
            if page.value == index:
                return page
        raise ValueError(f"No ActivePage with index {index}")

    @classmethod
    def from_name(cls, name: str) -> 'ActivePage':
        for page in cls:
            if page.name == name:
                return page
        raise ValueError(f"No ActivePage with name {name}")

    @classmethod
    def value_to_string(cls, value: int) -> str:
        mapping = {
            0: 'Intravascular',
            1: 'CCTA',
        }
        return mapping.get(value, 'Unknown')

    @classmethod
    def metadata_type(cls, page: 'ActivePage') -> object | str:
        mapping = {
            cls.INTRAVASCULAR: MetaDataIntravascular(),
            cls.CCTA: MetaDataCCTA(),
        }
        return mapping.get(page, 'unknown_metadata')
