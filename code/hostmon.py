"""CSV process-tree RSS (sum; shared pages can be counted more than once)."""
import argparse
import csv
import time
from common import utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid-tree',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    import psutil
    with open(a.out,'w') as out:
        writer=csv.writer(out);writer.writerow(['utc','pid','rss_bytes'])
        while True:
            try:
                parent=psutil.Process(a.pid_tree);processes=[parent]+parent.children(recursive=True)
                rss=0
                for process in processes:
                    try:rss+=process.memory_info().rss
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                writer.writerow([utc(),a.pid_tree,rss]);out.flush();time.sleep(1)
            except psutil.NoSuchProcess:break


if __name__=='__main__':main()
