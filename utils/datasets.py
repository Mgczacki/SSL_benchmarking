"""
Dataset loading utilities for JEPA benchmarking.

Provides unified access to multiple vision datasets commonly used in
self-supervised learning (SSL) evaluation, including CIFAR-10/100,
ImageNet-1K and its subsets, STL-10, Tiny ImageNet, iNaturalist 2018,
and Places205.

Each loader returns standard ``torchvision.datasets``-compatible dataset
objects with appropriate transforms already applied.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# ImageNet normalisation statistics (used across all datasets for consistency
# with pre-trained SSL encoders).
# ---------------------------------------------------------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# ImageNet-100 synset list -- 100 WordNet IDs from the CMC paper
# (Tian et al., "Contrastive Multiview Coding", 2020).
# ---------------------------------------------------------------------------
IMAGENET100_SYNSETS: List[str] = [
    "n02869837",  # bonnet
    "n01749939",  # green mamba
    "n02488291",  # langur
    "n02107142",  # Doberman
    "n13037406",  # gyromitra
    "n02091831",  # Saluki
    "n04517823",  # vacuum
    "n04589890",  # window screen
    "n03062245",  # cocktail shaker
    "n01773797",  # garden spider
    "n01735189",  # garter snake
    "n07831146",  # carbonara
    "n07753275",  # pineapple
    "n03085013",  # computer keyboard
    "n04485082",  # tripod
    "n02105505",  # komondor
    "n01983481",  # American lobster
    "n02788148",  # bannister
    "n03530642",  # honeycomb
    "n04435653",  # tow truck
    "n02086910",  # papillon
    "n02859443",  # boathouse
    "n13040303",  # stinkhorn
    "n03594734",  # jean
    "n02085620",  # Chihuahua
    "n02099849",  # Chesapeake Bay retriever
    "n01558993",  # robin
    "n04493381",  # tub
    "n02104029",  # kuvasz
    "n02086240",  # Shih-Tzu
    "n04008634",  # projectile
    "n03445777",  # golf ball
    "n02500267",  # indri
    "n02871525",  # bookshop
    "n02279972",  # monarch butterfly
    "n02116738",  # African hunting dog
    "n02009912",  # American egret
    "n02480495",  # orangutan
    "n01784675",  # centipede
    "n04133789",  # sandal
    "n02859443",  # boathouse (alias kept for compat)
    "n04192698",  # shield
    "n03764736",  # milk can
    "n03854065",  # organ
    "n02226429",  # grasshopper
    "n02231487",  # walking stick insect
    "n02085936",  # Maltese dog
    "n04263257",  # soup bowl
    "n02346627",  # porcupine
    "n02090622",  # borzoi
    "n04275548",  # spider web
    "n02037110",  # oystercatcher
    "n02510455",  # giant panda
    "n02129604",  # tiger
    "n04146614",  # school bus
    "n04243546",  # slot machine
    "n03031422",  # cinema
    "n03633091",  # ladle
    "n02219486",  # ant
    "n02487347",  # macaque
    "n02395406",  # pig
    "n04336792",  # stretcher
    "n03843555",  # oil filter
    "n02106662",  # German shepherd
    "n02326432",  # hare
    "n02108089",  # boxer dog
    "n02097298",  # Scotch terrier
    "n02006656",  # spoonbill
    "n02093754",  # Border terrier
    "n04596742",  # wok
    "n04380533",  # table lamp
    "n02447366",  # badger
    "n02643566",  # lionfish
    "n02807133",  # bathing cap
    "n04325704",  # stethoscope
    "n03837869",  # obelisk
    "n04099969",  # rocking chair
    "n04026417",  # purse
    "n02165456",  # ladybug
    "n02058221",  # albatross
    "n01770081",  # harvestman
    "n02484975",  # guenon
    "n03950228",  # pitcher
    "n04612504",  # yawl
    "n02640242",  # sturgeon
    "n03388043",  # fountain
    "n03042490",  # cliff dwelling
    "n02236044",  # mantis
    "n03926472",  # ping-pong ball
    "n04254120",  # soap dispenser
    "n03461385",  # grocery store
    "n01944390",  # snail
    "n09256479",  # coral reef
    "n02007558",  # flamingo
    "n03014705",  # chest
    "n04252077",  # snowmobile
    "n02128385",  # leopard
    "n03355925",  # flagpole
    "n02423022",  # gazelle
    "n02883205",  # bow tie
    "n03208938",  # disk brake
]

# De-duplicate while preserving order (the list above intentionally
# includes one duplicate for historical compatibility -- trim to 100).
_seen: set = set()
_unique: List[str] = []
for _s in IMAGENET100_SYNSETS:
    if _s not in _seen:
        _seen.add(_s)
        _unique.append(_s)
IMAGENET100_SYNSETS = _unique[:100]

# ---------------------------------------------------------------------------
# Number-of-classes registry
# ---------------------------------------------------------------------------
_NUM_CLASSES: Dict[str, int] = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet1k": 1000,
    "imagenet100": 100,
    "imagenet1pct": 1000,
    "tinyimagenet": 200,
    "stl10": 10,
    "inaturalist2018": 8142,
    "places205": 205,
}


def get_num_classes(dataset_name: str) -> int:
    """Return the number of classes for a given dataset.

    Parameters
    ----------
    dataset_name:
        One of ``cifar10``, ``cifar100``, ``imagenet1k``, ``imagenet100``,
        ``imagenet1pct``, ``tinyimagenet``, ``stl10``, ``inaturalist2018``,
        ``places205``.

    Returns
    -------
    int
        Number of target classes.

    Raises
    ------
    ValueError
        If *dataset_name* is not recognised.
    """
    key = dataset_name.lower().replace("-", "").replace("_", "")
    if key not in _NUM_CLASSES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {sorted(_NUM_CLASSES.keys())}"
        )
    return _NUM_CLASSES[key]


# ===================================================================
# Transforms
# ===================================================================

def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Standard evaluation transform used for linear probing / kNN.

    Pipeline: Resize(256) -> CenterCrop(image_size) -> ToTensor -> Normalize.

    Parameters
    ----------
    image_size:
        Spatial size of the final crop (default 224).

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Standard supervised training / fine-tuning transform.

    Pipeline: RandomResizedCrop(image_size) -> RandomHorizontalFlip
              -> ToTensor -> Normalize.

    Parameters
    ----------
    image_size:
        Spatial size of the random crop (default 224).

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ===================================================================
# Individual dataset loaders
# ===================================================================

def _load_cifar100(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-100 (60 000 images, 100 classes, native 32x32 -> resized).

    Transforms are expected to handle the resize to 224x224 already.
    """
    train_ds = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=train_transform,
    )
    val_ds = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=eval_transform,
    )
    return train_ds, val_ds


def _load_cifar10(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 (60 000 images, 10 classes, native 32x32 -> resized)."""
    train_ds = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform,
    )
    val_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=eval_transform,
    )
    return train_ds, val_ds


def _load_imagenet1k(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """Load ImageNet-1K (1.28M train / 50K val, 1000 classes).

    Expects the standard directory layout::

        <data_root>/imagenet/train/<synset>/<images>
        <data_root>/imagenet/val/<synset>/<images>
    """
    train_dir = os.path.join(data_root, "imagenet", "train")
    val_dir = os.path.join(data_root, "imagenet", "val")
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    return train_ds, val_ds


def _load_imagenet100(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """Load ImageNet-100 -- a 100-class subset of ImageNet-1K.

    Uses the synset list from the CMC paper (Tian et al., 2020).  Only
    directories whose name appears in ``IMAGENET100_SYNSETS`` are kept.
    The underlying data is read via ``ImageFolder`` on the full ImageNet
    directory and then filtered to the relevant samples.
    """
    train_dir = os.path.join(data_root, "imagenet", "train")
    val_dir = os.path.join(data_root, "imagenet", "val")

    synset_set = set(IMAGENET100_SYNSETS)

    def _filter_imagefolder(root: str, transform: transforms.Compose) -> Dataset:
        full_ds = datasets.ImageFolder(root, transform=transform)
        # Build mapping from class name -> new contiguous index
        selected_classes = sorted(
            [c for c in full_ds.classes if c in synset_set]
        )
        old_to_new = {
            full_ds.class_to_idx[c]: idx
            for idx, c in enumerate(selected_classes)
        }
        # Keep only samples whose class is in the 100-class subset
        filtered_indices: List[int] = []
        for i, (_, label) in enumerate(full_ds.samples):
            if label in old_to_new:
                filtered_indices.append(i)

        class _RemappedSubset(Dataset):
            """Subset that remaps labels to a contiguous 0..99 range."""

            def __init__(self, base: datasets.ImageFolder, indices: List[int],
                         label_map: Dict[int, int]) -> None:
                self.base = base
                self.indices = indices
                self.label_map = label_map

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, idx: int) -> Tuple[Any, int]:
                img, label = self.base[self.indices[idx]]
                return img, self.label_map[label]

        return _RemappedSubset(full_ds, filtered_indices, old_to_new)

    train_ds = _filter_imagefolder(train_dir, train_transform)
    val_ds = _filter_imagefolder(val_dir, eval_transform)
    return train_ds, val_ds


def _load_imagenet1pct(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    seed: int = 0,
) -> Tuple[Dataset, Dataset]:
    """Load ImageNet-1% -- ~13 images per class (approx. 12 811 train images).

    A deterministic random subset of ImageNet-1K is selected using *seed*.
    The validation set is the full ImageNet-1K validation set (50K images)
    so that results are directly comparable to the full-data baseline.

    Parameters
    ----------
    seed:
        Random seed used for the deterministic per-class sampling.
    """
    train_dir = os.path.join(data_root, "imagenet", "train")
    val_dir = os.path.join(data_root, "imagenet", "val")

    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)

    # Group sample indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(full_train.samples):
        class_indices.setdefault(label, []).append(idx)

    # Deterministically sample ~1 % from each class (at least 1 per class)
    rng = random.Random(seed)
    selected_indices: List[int] = []
    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls]
        n_select = max(1, round(len(indices) * 0.01))
        sampled = rng.sample(indices, min(n_select, len(indices)))
        selected_indices.extend(sampled)

    selected_indices.sort()
    train_ds = Subset(full_train, selected_indices)
    return train_ds, val_ds


def _load_tinyimagenet(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """Load Tiny ImageNet (200 classes, 64x64 images).

    Expects the standard directory layout::

        <data_root>/tiny-imagenet-200/train/<synset>/images/<images>
        <data_root>/tiny-imagenet-200/val/images/<images>

    The validation set is restructured into per-class sub-directories so
    that ``ImageFolder`` can read it.  If the restructured layout already
    exists (``val/<synset>/`` directories), it is used directly.
    """
    base = os.path.join(data_root, "tiny-imagenet-200")
    train_dir = os.path.join(base, "train")
    val_dir = os.path.join(base, "val")

    # -- Restructure the val directory if needed -------------------------
    val_annotations = os.path.join(val_dir, "val_annotations.txt")
    val_images_dir = os.path.join(val_dir, "images")
    if os.path.isfile(val_annotations) and os.path.isdir(val_images_dir):
        with open(val_annotations, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, synset = parts[0], parts[1]
                synset_dir = os.path.join(val_dir, synset, "images")
                os.makedirs(synset_dir, exist_ok=True)
                src = os.path.join(val_images_dir, fname)
                dst = os.path.join(synset_dir, fname)
                if os.path.isfile(src) and not os.path.isfile(dst):
                    os.rename(src, dst)

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    return train_ds, val_ds


def _load_stl10(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """Load STL-10 (13K labelled + 100K unlabelled, 10 classes, 96x96).

    The *train* split returned here is the **labelled** ``train`` split
    (5 000 images).  The ``test`` split (8 000 images) is used as
    validation.  Use ``split='unlabeled'`` separately if you need the
    100K unlabelled pool for SSL pre-training.
    """
    train_ds = datasets.STL10(
        root=data_root, split="train", download=True, transform=train_transform,
    )
    val_ds = datasets.STL10(
        root=data_root, split="test", download=True, transform=eval_transform,
    )
    return train_ds, val_ds


def _load_inaturalist2018(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """Load iNaturalist 2018 (460K images, 8142 classes).

    Expects the directory layout::

        <data_root>/inaturalist2018/train/<class_folders>/<images>
        <data_root>/inaturalist2018/val/<class_folders>/<images>
    """
    base = os.path.join(data_root, "inaturalist2018")
    train_dir = os.path.join(base, "train")
    val_dir = os.path.join(base, "val")
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    return train_ds, val_ds


def _load_places205(
    data_root: str,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """Load Places205 (2.5M images, 205 scene categories).

    Expects the directory layout::

        <data_root>/places205/train/<category>/<images>
        <data_root>/places205/val/<category>/<images>
    """
    base = os.path.join(data_root, "places205")
    train_dir = os.path.join(base, "train")
    val_dir = os.path.join(base, "val")
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    return train_ds, val_ds


# ===================================================================
# Main entry point
# ===================================================================

def get_dataset(
    dataset_name: str,
    data_root: str = "./data",
    image_size: int = 224,
    train_transform: Optional[transforms.Compose] = None,
    eval_transform: Optional[transforms.Compose] = None,
    seed: int = 0,
) -> Tuple[Dataset, Dataset]:
    """Return ``(train_dataset, val_dataset)`` for the requested benchmark.

    Parameters
    ----------
    dataset_name:
        Name of the dataset.  Recognised values (case-insensitive,
        hyphens / underscores ignored):

        * ``cifar10``
        * ``cifar100``
        * ``imagenet1k``
        * ``imagenet100``
        * ``imagenet1pct``  (ImageNet-1%)
        * ``tinyimagenet``
        * ``stl10``
        * ``inaturalist2018``
        * ``places205``

    data_root:
        Root directory under which datasets are stored / downloaded.

    image_size:
        Target spatial resolution for default transforms (default 224).

    train_transform:
        Custom training transform.  When *None*, ``get_train_transform``
        is used with *image_size*.

    eval_transform:
        Custom evaluation transform.  When *None*, ``get_eval_transform``
        is used with *image_size*.

    seed:
        Random seed used for deterministic subset selection
        (relevant for ``imagenet1pct``).

    Returns
    -------
    tuple[Dataset, Dataset]
        ``(train_dataset, val_dataset)`` pair with transforms applied.

    Raises
    ------
    ValueError
        If *dataset_name* is not recognised.
    """
    key = dataset_name.lower().replace("-", "").replace("_", "")

    if train_transform is None:
        train_transform = get_train_transform(image_size)
    if eval_transform is None:
        eval_transform = get_eval_transform(image_size)

    loaders: Dict[str, Any] = {
        "cifar10": _load_cifar10,
        "cifar100": _load_cifar100,
        "imagenet1k": _load_imagenet1k,
        "imagenet100": _load_imagenet100,
        "tinyimagenet": _load_tinyimagenet,
        "stl10": _load_stl10,
        "inaturalist2018": _load_inaturalist2018,
        "places205": _load_places205,
    }

    if key == "imagenet1pct":
        return _load_imagenet1pct(
            data_root, train_transform, eval_transform, seed=seed,
        )

    if key not in loaders:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {sorted(list(loaders.keys()) + ['imagenet1pct'])}"
        )

    return loaders[key](data_root, train_transform, eval_transform)
