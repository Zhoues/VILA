__all__ = ["Media", "File", "Image", "Video", "Spatial"]


class Media:
    pass


class File(Media):
    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    pass


class Video(File):
    pass

# NOTE(Zhouenshen): Spatial is a special file that contains spatial information
class Spatial:
    def __init__(self, spatial_feature):
        self.spatial_feature = spatial_feature
