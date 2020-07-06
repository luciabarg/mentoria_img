import argparse

import os

from tqdm import tqdm

from torch import nn
from torchvision import models

from torchvision.transforms import Compose, Resize, Normalize, ToTensor

from PIL import Image

import json


def progressbar(x, **kwargs):
    return tqdm(x, ascii=True, **kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Feature extraction",
        add_help=True,
        allow_abbrev=False
    )

    parser.add_argument(
        "root",
        help="path to the images",
        type=str,
    )

    parser.add_argument(
        "--output-file",
        help="output file",
        type=str,
        default="features.json"
    )

    return parser.parse_args()


def main(args):
    # get file list
    root = os.path.abspath(args.root)
    file_list = []
    for root, _, files in os.walk(root, followlinks=True):
        valid_files = [f for f in files
                       if os.path.splitext(f)[1] in (".jpg", ".jpeg", ".png")]
        file_list += [os.path.join(root, f) for f in valid_files]

    print(f"{len(file_list)} images")

    # preprocessing and normalization
    preprocess = Compose([
        Resize(size=(224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
        ])

    # load pretrained model and discard last fc layer
    model = models.resnet101(pretrained=True)
    modules = list(model.children())[:-1]
    extractor = nn.Sequential(*modules)
    extractor.eval()

    # extract features
    output_dict = {}
    for f in progressbar(file_list):
        img = Image.open(f)
        img = preprocess(img)
        img = img.unsqueeze(0)

        feature = extractor(img)
        feature = feature.squeeze()

        output_dict[f] = feature.detach().tolist()

    with open(args.output_file, "w") as fh:
        json.dump(output_dict, fh)
        print(f"{args.output_file} saved ({len(output_dict)} features)")


if __name__ == '__main__':
    main(parse_arguments())
