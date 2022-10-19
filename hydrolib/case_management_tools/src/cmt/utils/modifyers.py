from pathlib import Path, PurePath


def path_modifyer(path, prefix, exclude_suffices=[], exclude_names=[]):
    if path is not None:
        if (path.suffix.lower() not in exclude_suffices) & (
            path.name not in exclude_names
        ):
            path = Path(prefix).joinpath(path)
    return path


def prefix_to_paths(item, prefix, exclude_suffices=[], exclude_names=[]):
    if type(item) == list:
        for idx, i in enumerate(item):
            item[idx] = prefix_to_paths(i, prefix, exclude_suffices, exclude_names)
    else:
        # we change a PurePath item ór an item with a filepath (=PurePath) property
        if isinstance(item, PurePath):
            item = path_modifyer(item, prefix, exclude_suffices, exclude_names)
        elif hasattr(item, "filepath"):
            item.filepath = path_modifyer(
                item.filepath, prefix, exclude_suffices, exclude_names
            )

    return item
