from PIL import ImageEnhance, ImageFilter


ALLOWED_TRANSFORMS = {
    "brightness": [0.4, 0.6, 0.8],
    "gaussian_blur": [1.0, 2.0, 3.0],
    "contrast": [0.5, 0.7, 1.3],
}


def apply_transform(image, transform_name, value):
    """
    Apply a deterministic transformation.

    Invalid transform names or parameters are rejected.
    """

    if transform_name not in ALLOWED_TRANSFORMS:
        raise ValueError(
            f"Unsupported transform: {transform_name}"
        )

    allowed_values = ALLOWED_TRANSFORMS[transform_name]

    if value not in allowed_values:
        raise ValueError(
            f"Invalid value {value} for {transform_name}. "
            f"Allowed: {allowed_values}"
        )

    if transform_name == "brightness":
        return ImageEnhance.Brightness(image).enhance(value)

    if transform_name == "gaussian_blur":
        return image.filter(
            ImageFilter.GaussianBlur(radius=value)
        )

    if transform_name == "contrast":
        return ImageEnhance.Contrast(image).enhance(value)

    raise ValueError("Unexpected transformation")
