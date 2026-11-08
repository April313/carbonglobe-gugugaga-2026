"""Run an archived script with a configurable workspace, without changing algorithms."""
import argparse,os,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('--workspace',default='.');args,extra=p.parse_known_args()
root=Path(__file__).resolve().parent;script=(root/args.script).resolve()
assert script.is_relative_to(root/'src') and script.suffix=='.py'
os.chdir(Path(args.workspace).resolve())
sys.path[:0]=[str(script.parent),str(root/'src/compat'),str(root/'src/s1')]
sys.argv=[str(script)]+extra
runpy.run_path(str(script),run_name='__main__')
