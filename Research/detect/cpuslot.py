# with slot(): ...   holds one of DR_CPU_SLOTS lock files while a heavy piece of work runs, so that several of these scripts
# started side by side use no more than that many cores between them. Not set (the default): no limit.
import fcntl, os, time
import paths
N = int(os.environ.get('DR_CPU_SLOTS', '0'))
class slot:
    def __enter__(self):
        self.f = None
        while N > 0:
            for i in range(1, N + 1):
                f = open(paths.WORK + 'tmp/slot%d.lock' % i, 'w')
                try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); self.f = f; return self
                except BlockingIOError: f.close()
            time.sleep(0.2)
        return self
    def __exit__(self, *a):
        if self.f is not None: fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()
