# IBW Preview Test

This folder keeps the manual AFM/PFM preview test workflow in one place:

- `sample.ibw`: sample input data.
- `ibw_preview.py`: script and pytest regression checks for the sample preview.
- `outputs/`: generated PNG previews.

Run from the AFM-tools repository root:

```powershell
python tests\ibw_preview\ibw_preview.py
```

Generate all channels:

```powershell
python tests\ibw_preview\ibw_preview.py --channels all
```

The default outputs are:

- `outputs/sample_ibw_preview_preferred.png`
- `outputs/sample_ibw_preview_all.png`

Run the sample-preview checks:

```powershell
pytest tests\ibw_preview\ibw_preview.py
```
