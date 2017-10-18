#! /usr/bin/env python
import task_manager as tm
import time

bset_sm = {'q':'caldera','m':'12000'};
bset_lg = {'q':'geyser','m':'300000'};


# stop script (graceful exit) and wait on jobs after:
tm.QUEUE_MAX_HOURS = 1./3600.

# max number of jobs in queue
tm.MAXJOBS = 50

# loop and submit
jobs = []
for i in range(1,3):
    jid = tm.submit(['sleep','10'])
    jobs.extend(jid)
    time.sleep(2)

# submit dependent job
print('dep job')
jid = tm.submit(['ls','-1'],depjob=jobs)

# wait on all jobs
ok = tm.wait()

# report status
print 'OK (False indicates an error): '+str(ok)  
