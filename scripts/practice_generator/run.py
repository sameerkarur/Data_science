#!/usr/bin/env python3
"""Regenerate all practice materials (50+ Q per subtopic)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from banks_course01 import COURSE_01_TOPICS
from banks_remaining import ALL_TOPICS
from core import generate_all

if __name__ == "__main__":
    generate_all([*COURSE_01_TOPICS, *ALL_TOPICS])
    print("Done. See PRACTICE_GUIDE.md for how to study.")
