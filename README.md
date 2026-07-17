# Modly GaussianGPT Extension

A Modly model extension for unconditional 3D Gaussian scene generation using
the official [GaussianGPT](https://github.com/nicolasvonluetzow/GaussianGPT)
implementation.

## Installation

1. In Modly, open **Models/Extensions → Install from GitHub**.
2. Install `https://github.com/DrHepa/modly-gaussiangpt-extension`.
3. Run **Setup** when prompted. Use **Repair** if the environment needs to be
   prepared again.
4. Open the Models UI and download the checkpoints for the node you want to
   use.

Setup installs the extension dependencies only. Model weights are downloaded
separately from the Modly UI and are never downloaded by `setup.py`.

## Requirements

A CUDA-capable NVIDIA GPU and enough storage for the selected checkpoint are
required. No additional platform compatibility is claimed.

## Nodes

| Node | Dataset | Input |
| --- | --- | --- |
| `gaussiangpt/generate-vfront` | 3D-FRONT | None |
| `gaussiangpt/generate-both` | 3D-FRONT and ASE | None |

GaussianGPT generates scenes unconditionally. The nodes do not accept prompts
or images.

## Parameters

- `seed`: random seed.
- `temperature`: sampling temperature.
- `top_p`: nucleus sampling threshold.
- `top_k`: top-k sampling limit; `0` disables it.
- `background_color`: preview background color.
- `render_preview`: enables or disables the orbit preview.

## Outputs

Each generation produces:

- `preview.glb`: point-based preview returned to Modly.
- `scene.pt`: GaussianGPT tensor data.
- `scene.ply`: Gaussian scene in PLY format.
- `orbit.gif`: optional orbit preview.
- `metadata.json`: generation metadata.

`preview.glb` is a compatibility preview, not a surface mesh or a complete
Gaussian-splat representation. Use `scene.pt` or `scene.ply` when the complete
Gaussian data is required.

## Credits

- Modly extension by [DrHepa](https://github.com/DrHepa).
- GaussianGPT by [Nicolas von Lützow](https://github.com/nicolasvonluetzow).

## License

The Modly extension wrapper is available under the [MIT License](LICENSE).
The vendored GaussianGPT source retains its separate
[MIT License](vendor/GaussianGPT/LICENSE). The checkpoint license is unknown
and is not granted by either source-code license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
