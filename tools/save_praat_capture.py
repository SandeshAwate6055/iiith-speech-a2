"""Persist approved Praat screenshot chunks only inside outputs/praat_screenshots."""
import argparse,base64,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('mode',choices=['start','append','finish','stream']);p.add_argument('chunk',nargs='?',default='');a=p.parse_args()
if not re.fullmatch(r'[a-zA-Z0-9_-]+\.(jpg|png)',a.name):raise ValueError('Invalid screenshot filename')
r=Path(__file__).resolve().parents[1]/'assignment_2a_mfa/outputs/praat_screenshots';r.mkdir(parents=True,exist_ok=True);target=r/a.name;temp=r/(a.name+'.partial')
if a.mode=='stream':
 import sys,io,ctypes,msvcrt
 from PIL import Image
 kernel=ctypes.windll.kernel32
 kernel.GetConsoleMode.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
 kernel.SetConsoleMode.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
 handle=msvcrt.get_osfhandle(sys.stdin.fileno());mode=ctypes.c_uint32()
 if kernel.GetConsoleMode(handle,ctypes.byref(mode)):kernel.SetConsoleMode(handle,mode.value & ~4)
 print('READY',flush=True)
 lines=[]
 for line in sys.stdin:
  if line.strip()=='ENDCAPTURE':break
  lines.append(line.strip())
 raw=base64.b64decode(''.join(lines),validate=True)
 with Image.open(io.BytesIO(raw)) as image:image.verify()
 with target.open('xb') as stream:stream.write(raw)
 print('Saved '+str(target),flush=True)
 raise SystemExit(0)
if a.mode=='finish':
 from PIL import Image
 with Image.open(temp) as image:image.verify()
 if target.exists():raise FileExistsError(target)
 temp.rename(target)
 print('Saved '+str(target))
else:
 with temp.open('xb' if a.mode=='start' else 'ab') as stream:stream.write(base64.b64decode(a.chunk,validate=True))
