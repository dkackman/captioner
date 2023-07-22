# Introduction

An experiment in using AI to manage and manipulate family photos.

Goals

- [x] Generically tag an entire directory strucutre of photos (a man and woman pose in front of a chirstmas tree)
- [ ] Train a model to recognize family members and pets
- [ ] Specifically caption an entire directory strucutre of photos (Don and Daisy pose in front of a chirstmas tree)
- [ ] Make an image search and organizer tool on top of that
- [ ] Create a stable diffusion LoRA from the specifically tagged photos (imagine a picutre of Don and Daisy in front of a christamas tree on USS Yorktown)

## Install

Make sure you have left the venv.

```bash
sh install.sh
. ./activate
```

## Scripts

### `01.prepare.py`

Copies image files from a known location, converting them all to jpg and filtering our thumbnails (by file size) and corrupt files. It also flatten any directroy structure and just gives every file a uuid name. (optional)

### `02.label.py`

Runs image-to-text cpationing on an directroy of photos. Stores the generated caption in a csv file.
