"""Reversible circuit blocks for the mixed-radix constructions.

All integer words are little-endian and simultaneously used registers must be
disjoint. Scratch starts in zero. Blocks documented as clean restore scratch;
blocks returning history or intermediate words require outer uncomputation.
No allocation operation resets an already occupied qubit.
"""
