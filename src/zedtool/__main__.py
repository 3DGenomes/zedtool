#!/usr/bin/env python3
import sys
from zedtool.cli import main, print_version

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: zedtool.py <config_file>")
        print_version()
        sys.exit(1)
    ret = main(sys.argv[1])
    print(ret)
    sys.exit(ret)
