_base_ = '_base_abinet-vision.py'

model = dict(
    decoder=dict(
        # --- insert your plate dictionary here ---
       dictionary=dict(
            type='Dictionary',
            dict_file='dicts/malaysia_plate.txt',
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=False,
            with_unknown=False
        ),
         d_model=512,
         num_iters=3,
         language_decoder=dict(
             type='ABILanguageDecoder',
             d_model=512,
             n_head=8,
             d_inner=2048,
             n_layers=4,
             dropout=0.1,
             detach_tokens=True,
             use_self_attn=False,
         )),
 )