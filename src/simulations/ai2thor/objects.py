"""AI2-THOR scene object helpers."""


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
