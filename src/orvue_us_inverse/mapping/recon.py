"""
orvue_us_inverse.mapping.recon - reconstruction from oracle labels (placeholder; implemented in S2).

S2: voxel grid from config.GridConfig; each frame pixel placed in 3D with T_measured and the probe geometry
(as BModeSimulator.plane_points), its label added to the nearest voxel's class histogram; voxel label = the
majority class, never-hit voxels stay unobserved; coverage count, mean-intensity volume and optional
filling of small enclosed gaps (never into unscanned areas).
"""
