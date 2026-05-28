import os
import glob
import hydra
import json

import pydicom as dcm
import SimpleITK as sitk
from dataclasses import asdict
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm

from version import version_file_str
from segmentation.predict import Predict
from segmentation.segment import mask_to_contours


@hydra.main(version_base=None, config_path='..', config_name='config')
def segment_files(config: DictConfig) -> None:
    input_dir = config.segmentation.input_dir
    files = glob.glob(input_dir + '/NARCO_*/Run*/*', recursive=True)
    files = [file for file in files if '_' not in os.path.basename(file)]  # exclude subdirs (all have _ in name)
    logger.info(f'Found {len(files)} files to segment')
    predictor = Predict(main_window=None, config=config)

    for file in tqdm(files, desc='Segmenting files', unit='files', leave=False):
        try:
            image = dcm.dcmread(os.path.join(input_dir, file), force=True).pixel_array
            if image.ndim == 4:  # 3 channel input
                image = image[:, :, :, 0]
        except (AttributeError, IsADirectoryError):
            try:  # NIfTi
                input_image = sitk.ReadImage(os.path.join(input_dir, file))
                image = sitk.GetArrayFromImage(input_image)
            except Exception:
                logger.info(f'Skipping file {file} as it is not a valid IVUS file (DICOM or NIfTi supported)')
                continue

        logger.info(f'Segmenting file {file}')
        lower_limit = 0
        upper_limit = image.shape[0]
        try:
            masks = predictor(image, lower_limit, upper_limit)
        except Exception:
            continue
        frame_data = mask_to_contours(None, masks, lower_limit, upper_limit, config=config)
        if frame_data is None:
            continue

        serializable = {str(i): asdict(fd) for i, fd in frame_data.items()}
        with open(os.path.join(input_dir, f'{file}_contours_{version_file_str}.json'), 'w') as out_file:
            json.dump(serializable, out_file)


if __name__ == '__main__':
    segment_files()
