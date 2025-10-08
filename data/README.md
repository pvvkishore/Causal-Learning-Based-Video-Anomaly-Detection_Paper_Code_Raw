# Example Annotation File Format

This file demonstrates the format for annotation files used in the video anomaly detection system.

## Format

The annotation file should be in JSON format with the following structure:

```json
{
  "video_filename.mp4": {
    "frame_index": label,
    ...
  },
  ...
}
```

Where:
- `video_filename.mp4`: The exact filename of the video file
- `frame_index`: Integer frame index (as a string)
- `label`: Binary label (0 = normal, 1 = anomaly)

## Example

```json
{
  "video001.mp4": {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 1,
    "4": 1,
    "5": 1,
    "6": 0
  },
  "video002.mp4": {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0
  }
}
```

## Notes

- If a frame index is not present in the annotations, it will be considered as normal (label 0)
- Frame indices start from 0
- Ensure video filenames match exactly (case-sensitive)
- All frame indices should be strings, not integers
