"""
orvue_us_inverse.mapping.poses - pose sources for a sweep (placeholder; implemented from S1).

S1: scripted sweep: serpentine lanes over the 100 x 100 mm region from SweepConfig (lane centres spread so the
    outer image edges touch the region edges, actual overlap >= requested), one pass per yaw, poses at the probe
    speed on simulated time.
S6: mouse pose source (hold-to-scan).  S7: camera-tracked pose source (tracking.tracker.ProbeTracker).
"""
