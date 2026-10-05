"""
Convert diffusion checkpoints saved before the GRIDAE -> VolumeVAE rename so they
load with the current LGCDiffTrainer / GraspDiffTrainer.

- state_dict keys 'grid_ae.*' are renamed to 'volume_vae.*'
- the saved hyperparameter ae.name 'GRIDAE*' is renamed to 'VolumeVAE*'

VolumeVAE checkpoints trained with LGTrainer store weights under 'model.' and
need no conversion.

Usage:
    python scripts/convert_legacy_ckpt.py <in.ckpt> [<out.ckpt>]
If <out.ckpt> is omitted, the result is written to <in>-volume_vae.ckpt.
"""
import argparse
import os.path as osp
import torch

OLD_PREFIX, NEW_PREFIX = 'grid_ae.', 'volume_vae.'
OLD_NAME, NEW_NAME = 'GRIDAE', 'VolumeVAE'


def convert(ckpt):
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {}
    cnt = 0
    for k, v in state_dict.items():
        if k.startswith(OLD_PREFIX):
            k = NEW_PREFIX + k[len(OLD_PREFIX):]
            cnt += 1
        new_state_dict[k] = v
    if 'state_dict' in ckpt:
        ckpt['state_dict'] = new_state_dict
    else:
        ckpt = new_state_dict
    print(f"Renamed {cnt} '{OLD_PREFIX}' keys to '{NEW_PREFIX}'.")

    hparams = ckpt.get('hyper_parameters', None) if 'state_dict' in ckpt else None
    ae_cfg = hparams.get('ae', None) if hparams is not None else None
    if ae_cfg is not None and str(ae_cfg.get('name', '')).startswith(OLD_NAME):
        old_name = ae_cfg['name']
        ae_cfg['name'] = NEW_NAME + old_name[len(OLD_NAME):]
        print(f"Renamed hyperparameter ae.name: {old_name} -> {ae_cfg['name']}")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='legacy checkpoint path')
    parser.add_argument('output', nargs='?', default=None, help='converted checkpoint path')
    args = parser.parse_args()

    output = args.output
    if output is None:
        root, ext = osp.splitext(args.input)
        output = f'{root}-volume_vae{ext}'

    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
    ckpt = convert(ckpt)
    torch.save(ckpt, output)
    print(f"Saved converted checkpoint to {output}")


if __name__ == '__main__':
    main()
