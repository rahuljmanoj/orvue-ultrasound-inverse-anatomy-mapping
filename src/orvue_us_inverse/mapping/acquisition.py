"""
orvue_us_inverse.mapping.acquisition - distance-triggered frame capture (placeholder; implemented in S1).

S1: takes poses from a pose source and captures a frame from the simulator (mapping.probe.make_simulator) every
frame_spacing_mm of travel or angle_trigger_deg of rotation; stores oracle labels (labels_image) and, with
AcquisitionConfig.store_images, the B-mode frame (render) as FrameRecords in a sweep_io.Sweep.
"""
