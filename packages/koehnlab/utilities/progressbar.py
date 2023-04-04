from collections.abc import Collection, Generator
from typing import TextIO
import sys


# Adapted from https://stackoverflow.com/a/34482761
def progressbar(
    collection: Collection, prefix: str = "", width: int = 60, out: TextIO = sys.stdout
) -> Generator:
    """Returns a generator iterating over the elements in the provided collection. While iterating
    over that generator, a progress bar tracking the iteration is printed to the given output stream
    (stdout by default)"""
    nEntries = len(collection)

    def show(j: int):
        covered = int(j * width / nEntries)
        print(
            f"{prefix}[{u'█' * covered}{('.' * (width - covered))}] {j}/{nEntries}",
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(collection):
        yield item
        show(i + 1)

    print("\n", flush=True, file=out)
