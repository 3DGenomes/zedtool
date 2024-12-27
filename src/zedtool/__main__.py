#!/usr/bin/env python3
import sys
from zedtool.cli import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: zedtools.py <config_file>")
        sys.exit(1)
    ret = main(sys.argv[1])
    print(ret)
    sys.exit(ret)
