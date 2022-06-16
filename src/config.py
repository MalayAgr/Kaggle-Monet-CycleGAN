import os

import torch


class Config:
    DATA_DIR = "../data"

    IMG_SIZE = 256

    # Used to ensure both Monet and normal images undergo the same transformation
    # See: https://albumentations.ai/docs/examples/example_multi_target/
    ADDITIONAL_TARGETS = {"monet_img": "image"}

    # Paper uses a batch size of 1
    BATCH_SIZE = 8

    EPOCHS = 200

    LR = 2e-4

    # This is a scaling factor for the cycle consistency
    # And the identity losses
    LAMBDA = 10

    VAL_SPLIT = 0.1

    # PyTorch device related arguments - gpu or cpu
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    PIN_MEMORY = True if torch.cuda.is_available() else False

    @classmethod
    def filepath(cls, filename="photo"):
        return os.path.join(cls.DATA_DIR, filename)
