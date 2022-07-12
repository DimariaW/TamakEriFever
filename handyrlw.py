# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import sys
import yaml
from handyrl_core.connection import *
from handyrl_core.worker import Worker, entry
import socket
from handyrl_core.environment import prepare_env

if __name__ == '__main__':
    with open('config.yaml') as f:
        args = yaml.safe_load(f)
    print(args)



    entry_args = args['entry_args']
    entry_args['host'] = socket.gethostname()

    args = entry(entry_args)
    print(args)
    prepare_env(args['env'])

    conn = connect_socket_connection(args['worker']['remote_host'], 9998)

    actor = Worker(args, conn, 0)
    actor.run()


