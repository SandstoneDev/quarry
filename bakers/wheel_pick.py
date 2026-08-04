"""wheel_pick - the wheel-node picker, split out of car_bake.py so the rule can be
unit-tested without pulling in the baker's external dependencies (core.imgarchive,
formats.carcols, gvcslib.*), which only exist inside the built converter bundle.
"""
WHEEL_ALIASES = ("wheel", "wheel1", "wheel_l")


def pick_wheel_node(frame_of):
    """The atomic holding the wheel MESH, or None when the model has no wheels.

 Measured across the 191 wheeled models on the disc: 173 call it "wheel", and the
 rest spread over wheel1, wheel_l/_r, wheel_front/_rear plus five spellings no list
 anticipated - wheel2 (buccanee, intruder, petrotr), wheel_rf (combine) and wheela
 (raindanc). Those five baked with no wheels at all.

 So: prefer the known names, then accept any other wheel* atomic. A _dummy suffix is
 excluded because that is a mount point, and a few models (Hydra's mid gear) give
 their mounts geometry. The ten wheel-less models - helicopters, boats, RC toys - carry no wheel* atomic whatsoever, so the fallback cannot mistake them for vehicles.
 """
    for alias in WHEEL_ALIASES:
        if alias in frame_of:
            return frame_of[alias]
    for name in sorted(frame_of):
        if name.startswith("wheel") and not name.endswith("_dummy"):
            return frame_of[name]
    return None
