# This code is derived from the library available at https://github.com/wking/igor,
# which is licensed under the GNU Lesser General Public License (LGPL) version 3.

# GNU LESSER GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007

#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.


"""Backward-compatible re-export of :mod:`.binarywave`.

The full Igor Binary Wave reader lives in :mod:`.binarywave`.  This module
exists only for code that still imports ``afm_tools.igor.igor``.  New code
should import ``afm_tools.igor.binarywave`` directly.
"""

from .binarywave import *  # noqa: F401, F403


def save(filename):
    """Save Igor Binary Wave — not implemented."""
    raise NotImplementedError
