import struct
import zlib

import pyarrow.parquet as pq

IMAGE_SIZE = 28


def _decode_grayscale_png(data: bytes) -> list[int]:
    """
    A minimal, hand-rolled decoder for the one PNG variant the bundled MNIST parquet files
    actually contain (confirmed via inspection): 8-bit grayscale, no interlacing, no palette.
    Uses only the stdlib zlib/struct modules - parsing PNG's chunk structure and un-filtering
    its scanlines is a small, well-specified algorithm, unlike parsing parquet itself (see
    the pyarrow dependency note in docs/structure.md's "MNIST" section).

    Returns a flat, row-major list of IMAGE_SIZE*IMAGE_SIZE pixel values (0-255).
    """

    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"

    pos = 8
    idat = b""
    width = height = bit_depth = color_type = None

    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 8 + length + 4  # skip the trailing CRC

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, _interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif chunk_type == b"IDAT":
            idat += chunk
        elif chunk_type == b"IEND":
            break

    assert bit_depth == 8 and color_type == 0, f"expected 8-bit grayscale; got bit_depth={bit_depth}, color_type={color_type}"

    raw = zlib.decompress(idat)
    stride = width
    pixels: list[int] = []
    previous_row = bytearray(stride)
    pos = 0

    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        row = bytearray(raw[pos : pos + stride])
        pos += stride

        for x in range(stride):
            a = row[x - 1] if x > 0 else 0
            b = previous_row[x]
            c = previous_row[x - 1] if x > 0 else 0

            if filter_type == 1:
                row[x] = (row[x] + a) & 0xFF
            elif filter_type == 2:
                row[x] = (row[x] + b) & 0xFF
            elif filter_type == 3:
                row[x] = (row[x] + (a + b) // 2) & 0xFF
            elif filter_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                row[x] = (row[x] + predictor) & 0xFF
            # filter_type == 0 (None): no change

        pixels.extend(row)
        previous_row = row

    return pixels


def load_mnist_dataset(path: str, limit: int | None = None) -> list[tuple[tuple[float, ...], int]]:
    """
    Parses one of the bundled MNIST parquet files (data/mnist/mnist-train.parquet or
    mnist-test.parquet - a HuggingFace-style export: an `image` struct column of PNG-encoded
    bytes, a `label` int64 column). Decodes each PNG, flattens row-major, normalizes each pixel
    to [0.0, 1.0] (divide by 255) - the same shape/normalization convention
    digits_data.load_digits_dataset already established for the smaller bundled dataset.

    limit caps how many rows are read (from the start of the file) - the real files are 60000/
    10000 rows, too many to decode in every test run; demo_mnist_recognition.py calls this with
    limit=None to use the real, full dataset.
    """

    table = pq.read_table(path)
    rows = table.slice(0, limit).to_pylist() if limit is not None else table.to_pylist()

    dataset: list[tuple[tuple[float, ...], int]] = []
    for row in rows:
        pixels = _decode_grayscale_png(row["image"]["bytes"])
        assert len(pixels) == IMAGE_SIZE * IMAGE_SIZE, f"expected a {IMAGE_SIZE}x{IMAGE_SIZE} image; got {len(pixels)} pixels"
        state = tuple(value / 255.0 for value in pixels)
        dataset.append((state, row["label"]))

    return dataset
