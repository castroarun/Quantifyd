"""Run a Python snippet on the VPS by writing to a temp file and executing it there."""
import sys
from _vps_helper import run, PW

snippet = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 60

# Upload snippet via heredoc cat
remote_path = f"/tmp/_vps_snippet_{abs(hash(snippet))}.py"

import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('94.136.185.54', username='arun', password=PW, timeout=15)

sftp = c.open_sftp()
with sftp.open(remote_path, 'w') as f:
    f.write(snippet)
sftp.close()

cmd = f"cd /home/arun/quantifyd && venv/bin/python3 {remote_path} && rm -f {remote_path}"
_, stdout, stderr = c.exec_command(cmd, timeout=timeout)
out = stdout.read().decode()
err = stderr.read().decode()
code = stdout.channel.recv_exit_status()
c.close()

print(out, end='')
if err:
    print('STDERR:', err, file=sys.stderr, end='')
sys.exit(code)
