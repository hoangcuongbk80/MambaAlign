import pickle
from enum import Enum
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from anomalib.data.utils import Split, LabelName, InputNormalizationMethod
from pandas import DataFrame

from datamodules.base import segmented2segmented, Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset


class NumSegmented(Enum):
    N0 = 0
    N16 = 16
    N53 = 53
    N126 = 126
    N246 = 246


def get_default_resolution():
    return 640, 232


def read_split(
    root: Path,
    num_segmented: NumSegmented,
    split: Split,
    supervision: Supervision,
) -> list[tuple[int, bool]]:
    fn = root / f"split_weakly_{num_segmented.value}.pyb"

    if split == Split.TEST:
        file_split = Split.TEST
    else:
        file_split = Split.TRAIN

    with open(fn, "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if file_split == "train":
            train_samples = list(
                map(
                    lambda sample: (
                        sample[0],
                        segmented2segmented(supervision, sample[1]),
                    ),
                    train_samples,
                )
            )
            return train_samples
        elif file_split == "test":
            # segmented and labeled always true
            test_samples = list(map(lambda sample: (sample[0], True), test_samples))
            return test_samples
        else:
            raise Exception(f"Unknown split {file_split}")


def is_mask_anomalous(path: str):
    img_arr = cv2.imread(path)
    if np.all(img_arr == 0):
        return LabelName.NORMAL
    return LabelName.ABNORMAL


class KSDD2Dataset(SSNDataset):
    """
    Dataset class for KolektorSDD2 dataset

    Args:
        root (Path): path to root of dataset
        supervision (Supervision): flag to signal the level of supervision for dataset
        transform (A.Compose): transforms used for preprocessing
        split (Split): either train or test split
        flips (bool): flag if dataset is extended by flipping (vert, horiz, 180).
        num_segmented (NumSegmented): number of segmented images in dataset
        dilate (int|None) if an int = size of dilation square, if None - not applied (default None)
        dt (tuple[int, int] | None) distance transform params (w, p), if None - not applied (default None)
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path,
        supervision: Supervision,
        transform: A.Compose,
        split: Split,
        flips: bool,
        normal_flips: bool,
        num_segmented: NumSegmented = NumSegmented.N0,
        dilate: int | None = None,
        dt: tuple[int, int] | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=flips,
            normal_flips=normal_flips,
            supervision=supervision,
            dilate=dilate,
            dt=dt,
            debug=debug,
        )
        self.num_segmented = num_segmented

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        # read the split with given number of segmented samples
        split_samples = read_split(
            self.root, self.num_segmented, self.split, self.supervision
        )

        # read into form "root, split, sample_id, image_path, mask_path" and only take samples that are segmented.
        # This enables us to have mixed supervised setup, while test remains fully segmented
        samples_list = [
            [
                str(self.root),
                sample_id,
                self.split.value,
                str(self.root / self.split.value / f"{sample_id}.png"),
                str(self.root / self.split.value / f"{sample_id}_GT.png"),
                is_segmented,
            ]
            for sample_id, is_segmented in split_samples
        ]
        samples = DataFrame(
            samples_list,
            columns=[
                "path",
                "sample_id",
                "split",
                "image_path",
                "mask_path",
                "is_segmented",
            ],
        )
        samples["label_index"] = samples["mask_path"].apply(is_mask_anomalous)
        samples.label_index = samples.label_index.astype(int)

        # add labels according to label index
        normal_samples = samples.loc[
            (samples.label_index == LabelName.NORMAL)
        ].reset_index()
        anomalous_samples = samples.loc[
            (samples.label_index == LabelName.ABNORMAL)
        ].reset_index()

        return normal_samples, anomalous_samples


class KSDD2(SSNDataModule):
    """
    Datamodule for KolektorSDD2

    Args:
        root (Path): path to root of dataset
        image_size ( int | tuple[int, int] | None): image size in format of (h, w)
        normalization (str | InputNormalizationMethod): normalization method for images, defaults to imagenet
        train_batch_size (int): batch size used in training
        eval_batch_size (int): batch size used in test / inference
        num_workers (int): number of dataloader workers. Must be <= 1 for supervised
        seed (int | None): seed
        flips (bool): flag if dataset is extended by flipping (vert, horiz, 180).
        num_segmented (NumSegmented): number of segmented images in dataset
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path | str,
        supervision: Supervision | None = None,
        image_size: tuple[int, int] | None = None,
        normalization: str
        | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        flips: bool = False,
        normal_flips: bool = False,
        num_segmented: NumSegmented = NumSegmented.N246,
        dilate: int | None = None,
        dt: tuple[int, int] | None = None,
        debug: bool = False,
    ) -> None:
        if supervision is None:
            if num_segmented == NumSegmented.N0:
                supervision = Supervision.UNSUPERVISED
            else:
                supervision = Supervision.FULLY_SUPERVISED
            print(
                f"Dataset supervision parameter is not set, it was automatically set to {supervision}"
            )
        print(f"Resolution set to: {image_size}")

        super().__init__(
            root=root,
            supervision=supervision,
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=flips,
        )

        self.train_data = KSDD2Dataset(
            transform=self.transform_train,
            split=Split.TRAIN,
            root=root,
            num_segmented=num_segmented,
            flips=flips,
            normal_flips=normal_flips,
            supervision=supervision,
            dilate=dilate,
            dt=dt,
            debug=debug,
        )
        self.test_data = KSDD2Dataset(
            transform=self.transform_eval,
            split=Split.TEST,
            root=root,
            num_segmented=num_segmented,
            flips=flips,
            normal_flips=False,
            supervision=supervision,
            debug=debug,
        )
