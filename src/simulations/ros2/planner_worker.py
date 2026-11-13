"""Stdlib-only ROS-side client for the persistent MARCO process."""
import json
import os
import select
import subprocess
import sys
import time


class PlannerWorker:
    def __init__(self, root, report, radius, planner_script=None):
        self.buffer=b'';self.ready=False;self.started=time.perf_counter()
        self.log=(report/'planner_worker.log').open('w')
        try:
            self.process=subprocess.Popen(
                [os.environ.get('MARCO_PYTHON', sys.executable),
                 str(planner_script or root/'three_robot_t/planner.py'),'--worker'],
                env={**os.environ,'DEMO_REPORT_DIR':str(report),
                     'DEMO_ROBOT_RADIUS':str(radius),'MARCO_GEOMETRY_BACKEND':'cpp'},
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
        except Exception:
            self.log.close();raise

    def read_reply(self, deadline):
        while b'\n' not in self.buffer:
            remaining=deadline-time.perf_counter()
            if remaining<=0:raise TimeoutError('MARCO worker reply timeout')
            if not select.select([self.process.stdout],[],[],remaining)[0]:
                raise TimeoutError('MARCO worker reply timeout')
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:raise RuntimeError('MARCO worker exited; see planner_worker.log')
            self.buffer+=chunk
        line,self.buffer=self.buffer.split(b'\n',1)
        return json.loads(line)

    def request(self, poses, preferred=False, timeout=45):
        started=time.perf_counter();deadline=started+timeout
        if not self.ready:
            if self.read_reply(deadline).get('ready') is not True:
                raise RuntimeError('Invalid MARCO worker startup reply')
            self.ready=True
        self.process.stdin.write((json.dumps({'poses':poses,'preferred':preferred})+'\n').encode())
        reply=self.read_reply(deadline)
        if not reply.get('ok'):raise RuntimeError('MARCO worker: '+reply.get('error','unknown error'))
        return {**reply['timings'],'rpc_wall_s':time.perf_counter()-started,
                'mode':'persistent_worker'}

    def close(self):
        try:
            if self.process.poll() is None:
                try:self.process.stdin.write(b'{"shutdown":true}\n')
                except (BrokenPipeError,OSError):pass
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill();self.process.wait()
        finally:
            self.process.stdin.close();self.process.stdout.close();self.log.close()
