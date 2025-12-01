import os
import tempfile
import sys


import sys
sys.path.append(os.path.abspath('./modules'))

import builtins
import datetime
import argparse

from modules.pe3r.demo import main_demo
from modules.pe3r.models import Models

def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print
    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now()
        formatted_date_time = now.strftime(time_format)
        builtin_print(f'[{formatted_date_time}] ', end='')  # print with time stamp
        builtin_print(*args, **kwargs)
    builtins.print = print_with_timestamp

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."), default=None)
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False, help="silence logs")
    # change defaults
    parser.prog = 'pe3r demo'
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path
        
    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    pe3r = Models(device=args.device)

    with tempfile.TemporaryDirectory(suffix='pe3r_gradio_demo') as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, pe3r, args.device, server_name, args.server_port, silent=args.silent)
