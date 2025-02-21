from torchvision import transforms

# Define ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard image size for resizing
IMAGE_SIZE = 224


def get_training_transform() -> transforms.Compose:
    """
    Compose a series of transformations for training images.

    The transformation pipeline includes:
      - Converting the image to a tensor.
      - Resizing the image to a fixed size.
      - Random horizontal flipping.
      - Normalization using ImageNet statistics.
      - Random cropping with padding.
      - Random erasing with a specified scale.

    Returns:
        transforms.Compose: A composition of training transformations.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE), padding=(8, 8)),
        transforms.RandomErasing(scale=(0.02, 0.4), value=IMAGENET_MEAN),
    ])
    return transform


def get_test_transform(equalized: bool = False) -> transforms.Compose:
    """
    Compose a series of transformations for test images.

    If `equalized` is True, the transformation pipeline will include random
    equalization; otherwise, it applies the standard transformations.

    Args:
        equalized (bool): Whether to apply random equalization. Default is False.

    Returns:
        transforms.Compose: A composition of test transformations.
    """
    if equalized:
        transform = transforms.Compose([
            transforms.RandomEqualize(p=1),
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transform
