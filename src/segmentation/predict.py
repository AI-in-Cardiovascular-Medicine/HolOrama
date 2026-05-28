import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QProgressDialog, QApplication
import gc


class Predict:
    def __init__(self, main_window, config=None) -> None:
        self.main_window = main_window
        config = main_window.config if config is None else config
        self.model_file = config.segmentation.model_file
        self.model_fold = config.segmentation.model_fold
        self.normalize = config.segmentation.normalize
        self.batch_size = config.segmentation.batch_size
        self.conserve_memory = config.segmentation.conserve_memory
        self.images = None

    def __call__(self, images, lower_limit, upper_limit) -> np.ndarray | None:
        self.images = images
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.normalisation(self.normalize)
        mask = self.inference()

        return mask

    def normalisation(self, do_it_or_not: bool):
        """Min-max normalisation of the images"""
        if do_it_or_not and self.images is not None:
            self.images = (self.images - self.images.min(axis=(1, 2), keepdims=True)) / (
                self.images.max(axis=(1, 2), keepdims=True) - self.images.min(axis=(1, 2), keepdims=True)
            )

    def inference(self):
        assert self.images is not None
        if "nnUNetTrainer" not in self.model_file:
            import tensorflow as tf

            custom_objects = {'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy}
            model = tf.keras.models.load_model(self.model_file, custom_objects=custom_objects, compile=False)

            self.check_input_shape(
                model.input_shape,
            )
            mask = np.zeros_like(self.images)

            if self.conserve_memory:
                if self.main_window is not None:
                    progress = QProgressDialog(
                        f'Please wait, segmenting frames {self.lower_limit + 1} to {self.upper_limit + 1}...',
                        'Cancel',
                        self.lower_limit,
                        self.upper_limit,
                        self.main_window,
                    )
                    progress.setWindowTitle('Automatic segmentation')
                    progress.setMinimumDuration(0)
                    progress.setModal(True)
                    progress.setValue(self.lower_limit)
                    QApplication.processEvents()
                    QApplication.processEvents()
                else:
                    progress = None

                for frame in range(self.lower_limit, self.upper_limit, self.batch_size):
                    if progress is not None:
                        progress.setValue(frame)
                        QApplication.processEvents()
                    # calling model() instead of model.predict() leads to smaller memory leak
                    pred = model(self.images[frame : frame + self.batch_size, :, :], training=False)
                    mask[frame : frame + self.batch_size, :, :] = np.array(pred)[0, :, :, :, 0]
                    if progress is not None and progress.wasCanceled():
                        progress.close()
                        return None
                if progress is not None:
                    progress.close()
            else:
                prediction = model.predict(
                    self.images[self.lower_limit : self.upper_limit, :, :], batch_size=self.batch_size, verbose=1
                )
                mask[self.lower_limit : self.upper_limit, :, :] = np.array(prediction)[0, :, :, :, 0]
        else:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            seg_predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device(device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True,
            )
            # initializes the network architecture, loads the checkpoint
            seg_predictor.initialize_from_trained_model_folder(
                self.model_file,
                use_folds=(self.model_fold,),
                checkpoint_name="checkpoint_final.pth",
            )
            print(f"Shape: {self.images.shape}")
            # mask = seg_predictor.predict_from_list_of_npy_arrays([img[None, None, ...] for img in self.images],
            #                                               segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
            #                                               properties_or_list_of_properties=[dict(spacing=[1, 1, 1]) for _ in self.images],
            #                                               truncated_ofname=None,
            #                                               num_processes=1)
            mask = seg_predictor.predict_single_npy_array(
                self.images[None, ...].astype(np.float32), image_properties=dict(spacing=[1, 1, 1])
            )
            print(f"mask shape: {mask.shape}")
        return mask

    def check_input_shape(self, input_shape, batch_size=16):
        """
        Check if the input shape of the model matches the shape of the images.
        model.input_shape

        """
        assert self.images is not None
        import tensorflow as tf

        logger.info(f"Input shape: {self.images.shape}")
        if input_shape[1] != self.images.shape[1] or input_shape[2] != self.images.shape[2]:
            logger.warning("Reshaping the images to match the model input shape.")

            # Ensure images are in float32 to save memory
            self.images = self.images.astype(np.float32)

            # Process images in batches to reduce memory usage
            num_images = self.images.shape[0]
            reshaped_images = []

            for start in range(0, num_images, batch_size):
                end = min(start + batch_size, num_images)
                batch = self.images[start:end]
                batch = np.expand_dims(batch, axis=-1)
                batch = tf.image.resize_with_crop_or_pad(batch, input_shape[1], input_shape[2])
                # batch = resize_with_crop_or_pad(batch, model.input_shape[1], model.input_shape[2])
                batch = np.squeeze(batch, axis=-1)
                reshaped_images.append(batch)

                # Explicitly call garbage collection
                gc.collect()

            self.images = np.concatenate(reshaped_images, axis=0)
            gc.collect()
