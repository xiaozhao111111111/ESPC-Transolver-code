PT_CONFIG = dict(in_channels=140, order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
                 stride=(2, 2, 2), enc_depths=(2, 2, 2, 2), enc_channels=(32, 64, 128, 256),
                 enc_num_head=(2, 4, 8, 16), enc_patch_size=(128, 128, 128, 128),
                 dec_depths=(2, 2, 2), dec_channels=(64, 64, 128), dec_num_head=(4, 4, 8),
                 dec_patch_size=(128, 128, 128), drop_path=0.0, shuffle_orders=False,
                 enable_flash=False, enable_rpe=False)


GRID_SIZE = 0.16  # normalized-coordinate voxel, not a FEM mesh size
