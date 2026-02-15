#!/usr/bin/env python3
"""Entry point for Crossplay engine. Run from repo root: python run.py"""

import sys
import os

# Ensure the repo root is on the Python path so 'crossplay' package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossplay.game_manager import GameManager

if __name__ == "__main__":
    gm = GameManager()
    gm.run()
