"""AI2-THOR scene object helpers."""

from .controller import require_success


def get_scene_objects(event):
    """Return the current object metadata list."""
    return event.metadata.get("objects", [])


def get_addable_object_types(event):
    """List pickupable object types that InitialRandomSpawn can duplicate."""
    return sorted(
        {
            obj["objectType"]
            for obj in get_scene_objects(event)
            if obj.get("pickupable", False)
        }
    )


def find_object_ids(event, object_type=None):
    """Find object IDs, optionally filtering by an exact object type."""
    return [
        obj["objectId"]
        for obj in get_scene_objects(event)
        if object_type is None or obj["objectType"] == object_type
    ]


def add_objects(controller, object_type, count=1, random_seed=0):
    """Attempt to add copies of an existing pickupable object type."""
    if count < 1:
        raise ValueError("count must be at least 1")

    addable_types = get_addable_object_types(controller.last_event)
    if object_type not in addable_types:
        raise ValueError(
            f"Cannot add {object_type!r}. Available types: {addable_types}"
        )

    current_count = len(find_object_ids(controller.last_event, object_type))
    event = controller.step(
        action="InitialRandomSpawn",
        randomSeed=random_seed,
        forceVisible=True,
        numPlacementAttempts=10,
        placeStationary=True,
        numDuplicatesOfType=[
            {"objectType": object_type, "count": current_count + count},
        ],
    )
    return require_success(event, f"Add {object_type}")


def disable_object(controller, object_id):
    """Temporarily hide an object so it can be enabled later."""
    event = controller.step(action="DisableObject", objectId=object_id)
    return require_success(event, f"Disable object {object_id}")


def enable_object(controller, object_id):
    """Restore an object previously hidden with disable_object."""
    event = controller.step(action="EnableObject", objectId=object_id)
    return require_success(event, f"Enable object {object_id}")


def remove_object(controller, object_id):
    """Hide an object at runtime using the reliable DisableObject action."""
    return disable_object(controller, object_id)


def remove_first_object_of_type(controller, object_type):
    """Hide the first runtime object matching an object type."""
    object_ids = find_object_ids(controller.last_event, object_type)
    if not object_ids:
        raise ValueError(f"No object of type {object_type!r} exists")
    return remove_object(controller, object_ids[0])


def remove_house_object(house, object_id):
    """Permanently delete an object from house JSON before controller creation."""

    def remove_from(objects):
        for index, obj in enumerate(objects):
            if obj["id"] == object_id:
                del objects[index]
                return True
            if remove_from(obj.get("children", [])):
                return True
        return False

    if not remove_from(house["objects"]):
        raise ValueError(f"No house object with id {object_id!r}")
    return house
