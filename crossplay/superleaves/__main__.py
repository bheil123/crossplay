"""
Entry point for: python -m crossplay.superleaves.trainer

IMPORTANT: This file exists to fix Windows multiprocessing.
On Windows, ProcessPoolExecutor uses 'spawn' which pickles functions
by module path. If trainer.py runs as __main__, its functions
(_init_worker, _play_batch) get __module__='__main__' and workers
can't find them. By importing from the actual module, the functions
keep their proper module path (crossplay.superleaves.trainer).
"""
from .trainer import main

if __name__ == '__main__':
    main()
