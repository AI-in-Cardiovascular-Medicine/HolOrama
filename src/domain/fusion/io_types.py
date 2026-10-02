from dataclasses import dataclass


@dataclass
class MetaDataFusion:
    modality: str = 'Fusion'
    patient_name: str = 'Unknown'
    birthdate: str = 'Unknown'
    sex: str = 'Unknown'
    ...
