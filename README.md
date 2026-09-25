# Skate 3 Texture Maker

Makes Skate 3 (PS3) `.psg` textures from any picture automatically, and turns
`.psg` textures back into PNGs so you can see and edit them.

## Quick start

1. Install [Python 3](https://www.python.org/downloads/) (tick "Add to PATH" on Windows).
2. Put your pictures (PNG, JPG, WEBP, GIF, BMP...) in the `input` folder.
3. Double-click `make_textures.bat` (Windows) or run `./make_textures.sh`.
4. Your textures are in the `output` folder.

The first run installs the two libraries it needs (Pillow and NumPy) by itself.

You can also **drag and drop** pictures or `.psg` files onto `make_textures.bat`:
pictures become `.psg`, and `.psg` files become `.png`.

## Replacing a texture in the game

A game texture is named by its ID (for example `106508011.psg`). To swap in
your own picture, give the output the same name:

```
python skate3tex.py my_deck.png --name 106508011
```

To edit an existing texture, turn it into a PNG, edit that, then convert it back:

```
python skate3tex.py 106508011.psg        # -> 106508011.png
python skate3tex.py 106508011.png        # -> 106508011.psg
```

## Options

| Option | What it does |
| --- | --- |
| `--fit cover` | Fill the square and crop whatever sticks out (default) |
| `--fit stretch` | Squash the whole picture into the square |
| `--fit contain` | Fit the whole picture and fill the empty borders with `--bg` |
| `--bg white` | Colour behind transparent areas (default `black`) |
| `--alpha 25` | Alpha value to store. The default is 25, the same as the original textures. `keep` keeps the picture's own transparency |
| `--name ID` | Output file name (one input only) |
| `-o DIR` | Output folder |
| `--preview` | Also save `NAME_preview.png` showing how the texture will look after compression |
| `--watch` | Keep running and convert anything you drop into `input` straight away |
| `--template X.psg` | Copy the file header from another texture |

## Format

Textures are RenderWare 4 PS3 (`RW4ps3`) files: 512×512, DXT5-compressed, with
10 mipmaps. That is a 568-byte header followed by 349,552 bytes of pixel data,
which comes to 350,120 bytes per file. The tool compresses the picture, builds
the mipmaps and writes the same header as the textures in this repo.
