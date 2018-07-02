import os
import signal
import webbrowser
from contextlib import redirect_stderr
from multiprocessing import Process
from tensorflow import app
from tensorboard import main

def launch_tb():
    def _f(r, w):
        os.close(r)
        with os.fdopen(w, 'w') as w, redirect_stderr(w):
            app.run(main.main, [None, "--logdir", "./graph"])

    r, w = os.pipe()
    p = Process(target=_f, args=(r, w))
    p.start()
    os.close(w)
    with os.fdopen(r) as r:
        r.readline()
    return p

if __name__ == "__main__":
    p = launch_tb()
    webbrowser.open_new_tab("http://127.0.0.1:6006")
    input("Press Enter to quit")
    os.kill(p.pid, signal.SIGINT)
