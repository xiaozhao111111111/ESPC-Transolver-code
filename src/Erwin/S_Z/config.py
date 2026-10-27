ERWIN_CONFIG = dict(c_in=140, c_hidden=[32, 64, 128, 256], enc_num_heads=[2, 4, 8, 16],
                    enc_depths=[2, 2, 2, 2], dec_num_heads=[2, 4, 8], dec_depths=[2, 2, 2],
                    strides=[2, 2, 2], ball_sizes=[32, 32, 32, 32],
                    rotate=45, dimensionality=3, mlp_ratio=4, mp_steps=0)
