#! /usr/bin/env python
from subprocess import Popen,PIPE,call
import os
import sys
from datetime import datetime
import time
import re
import tempfile
from glob import glob

#-- total runtime metrics
PROGRAM_START = datetime.now()  # init timer
QUEUE_MAX_HOURS = 20.           # trigger "stop" after QUEUE_MAX_HOURS

#-- job lists
JID = []           # the list of active job IDs
MAXJOBS = 400      # max number of jobs to keep in the queue
DEBUG_JOBS = True  # set to "True" to keep job.out files for jobs with "DONE" status

#--  ACCOUNT
ACCOUNT = 'NCGD0011'

#-- where to place output
JOB_LOG_DIR = '/glade/scratch/'+os.environ['USER']+'/lsf-calcs'
if not os.path.exists(JOB_LOG_DIR):
    stat = call(['mkdir','-p',JOB_LOG_DIR])
    if stat != 0: exit(1)

JOB_LOG_PREFIX = 'task_manger.calc'

#------------------------------------------------------------------------
#--- FUNCTION
#------------------------------------------------------------------------
def report_status(msg):
    '''
    print a status message with timestamp
    '''
    date = datetime.now()
    print(date.strftime("%Y %m %d %H:%M:%S")+': ' + msg)
    return

#------------------------------------------------------------------------
#--- FUNCTION
#------------------------------------------------------------------------
def total_elapsed_time():
    '''
    compute total time elapsed since initialization
    return time in hours
    '''
    return (datetime.now() - PROGRAM_START).total_seconds()/3600.

#------------------------------------------------------------------------
#--- FUNCTION
#------------------------------------------------------------------------
def _bwait(JOBID = [],njob_target=0):
    '''
    wait on a list of job IDs
    return when the number of running jobs has reached njob_target
    if DEBUG_JOBS = False, then delete .out file for jobs with DONE status
    '''

    ok = True

    if not JOBID:
        JOBID = JID

    #-- check total time
    stop_now = False
    if total_elapsed_time() > QUEUE_MAX_HOURS:
        report_status('total elapsed time: %.4f h'%total_elapsed_time())
        stop_now = True

    #-- check number of running jobs
    njob_running = len(JOBID)
    if njob_running <= njob_target:
        return ok,stop_now

    report_status('waiting on %d jobs'%njob_running)
    if njob_target == 0:
        print '-'*50
        for jobid in JOBID: print jobid,
        print
        print '-'*50
        print

    #-- wait on jobs
    status = {}
    first_run = True
    exit_list = []
    while (njob_running > njob_target):

        #-- loop over active jobs
        tmp = []
        for jobid in JOBID:

            #-- check status and report on first pass or if changed
            jobstat = _bstat(jobid)
            if njob_target == 0:
                if not first_run:
                    if not status[jobid] == jobstat:
                        report_status(jobid+' status: '+jobstat)
                else:
                    report_status(jobid+' status: '+jobstat)
            status[jobid] = jobstat

            #-- status dependent actions
            if any([jobstat == js for js in ['PEND','RUN','UNKWN']]):
                tmp.append(jobid)

            elif jobstat == 'DONE':
                if not DEBUG_JOBS:
                    jout = glob(os.path.join(
                            JOB_LOG_DIR,JOB_LOG_PREFIX+'.????????-??????.'+args[0]+'.out'))
                    if jout:
                        call(['rm','-f',jout[-1]])
            elif jobstat == 'EXIT':
                exit_list.append(jobid)
                ok = False
            elif jobstat == 'not found':
                #-- assume the job has completed successfully
                pass
            else:
                ok = False
                report_status(jobid+' unknown message: '+jobstat)

        #-- update list of active jobs to those still active
        JOBID[:] = tmp
        njob_running = len(tmp)
        del tmp[:]

        #-- finish loop
        first_run = False
        time.sleep(0.5)

    #-- update module variable
    JID[:] = JOBID

    #-- exit with messages
    if total_elapsed_time() > QUEUE_MAX_HOURS:
        stop_now = True

    if not ok:
        print
        print '-'*50
        report_status('Failed jobs:')
        for jobid in exit_list: print jobid,
        print
        print '-'*50
        print
    else:
        report_status('Done waiting.')

    if njob_running != 0:
        report_status('%d active jobs remain.'%njob_running)

    return ok,stop_now

#------------------------------------------------------------------------
#--- FUNCTION
#------------------------------------------------------------------------
def _bsub(cmdi,add_to_env={},depjob=[],bset_in={},job_name=''):
    '''
    submit a command line job to the queue
    '''
    ok = True
    stop = False

    #-- make sure command is a list not a string
    if type(depjob) is str:
        depjob = [depjob]

    #-- update environment with ammendments
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = 'no' # make python unbuffered
    if add_to_env:
        env.update(add_to_env)

    #-- submission settings
    bset = {'q' : 'geyser',
            'P' : ACCOUNT,
            'W' : '24:00'}
    if bset_in:
        bset.update(bset_in)

    if 'm' not in bset:
        if bset['q'] == 'geyser':
            bset['m'] = '350000'
        elif bset['q'] == 'caldera':
            bset['m'] = '20000'

    #-- compose submission string
    strcmdi = ' '.join(cmdi)

    date = datetime.now()
    if not job_name:
        job_name = cmdi[0]
    stdoe = JOB_LOG_DIR+'/'+JOB_LOG_PREFIX+'.'+date.strftime("%Y%m%d-%H%M%S")+'.%J.out'
    cmd = ['bsub',
           '-P',bset['P'],
           '-W',bset['W'],
           '-q',bset['q'],
           '-n','1',
           '-J',job_name,
           '-o',stdoe,
           '-e',stdoe]

    if 'm' in bset:
        cmd.extend(['-R','rusage[mem='+bset['m']+']'])
        cmd.extend(['-R','span[ptile=1]'])

    #-- add step to wait on dependent jobs
    if depjob:
        depjoblist = [jobid for jobid in depjob
                      if _bstat(jobid) != 'not found' or _bstat(jobid) != 'unknown']
        # this is the LSF way, but is not preferred because LSF forgets about
        # jobs in about an hour...so this fails for long running jobs
        #done_cond = ['done('+jobid+')'
        #             for jobid in depjob if _bstat(jobid) != 'not found']
        #cmd.extend(['-w',' && '.join(done_cond)])

        #-- add call to this script using bwait task in  __main__ section
        strd = __file__.replace('pyc','py')+' wait '+' '.join(depjoblist)+';'
        strcmdi = strd+' [ $? -eq 0 ] || exit 1 ; '+strcmdi

    cmd.append(strcmdi)

    #-- if number of jobs is at max, wait
    if len(JID) >= MAXJOBS:
        print 'Job count at threshold.'
        ok,stop = _bwait(JID,njob_target = MAXJOBS/2)
        if not ok:
            return None,ok,stop

    #-- submit the job
    p = Popen(cmd, stdin=None, stdout=PIPE, stderr=PIPE, env=env)
    stdout, stderr = p.communicate()

    #-- parse return string to get job ID
    try:
        jid = re.findall(r'\<(.*?)\>',stdout.strip())
        JID.append(jid[0])
    except:
        print 'BSUB failed:'
        print cmd
        print stdout
        print stderr
        return None,ok,stop

    #-- print job id and job submission string
    scmd = str(cmd)
    print '-'*50
    print jid[0]+': '+scmd
    print '-'*50
    print

    if total_elapsed_time() > QUEUE_MAX_HOURS:
        stop = True

    #-- return job id
    return [jid[0]],ok,stop

#------------------------------------------------------------------------
#--- FUNCTION
#------------------------------------------------------------------------
def _bjid(name):
    '''
    get job id from job name
    '''

    p = {}
    cmd = ['bjobs','-J',name]
    p[0] = Popen(cmd, stdin=None, stdout=PIPE, stderr=PIPE)

    cmd = ['tail','-n','+2']
    p[1] = Popen(cmd, stdin=p[0].stdout, stdout=PIPE, stderr=PIPE)

    cmd = ['awk','{print $1}']
    p[2] = Popen(cmd, stdin=p[1].stdout, stdout=PIPE, stderr=PIPE)

    stdout, stderr = p[2].communicate()

    jobid = list(stdout.strip().split('\n'))
    return jobid

#------------------------------------------------------------------------
#--- FUNCTION
#------------------------------------------------------------------------
def _bstat(jobid):
    '''
    return job status parsing bstat command
    '''
    p = {}
    cmd = ['bjobs','-a',str(jobid)]
    p[0] = Popen(cmd, stdin=None, stdout=PIPE, stderr=PIPE)

    cmd = ['grep','-o','not found']
    p[1] = Popen(cmd, stdin=p[0].stderr, stdout=PIPE, stderr=PIPE)

    stdout, stderr = p[1].communicate()
    if stdout:
        return stdout.strip()

    cmd = ['grep','-v','^    ']
    p[2] = Popen(cmd, stdin=p[0].stdout, stdout=PIPE, stderr=PIPE)

    cmd = ['tail','-n','+2']
    p[2] = Popen(cmd, stdin=p[2].stdout, stdout=PIPE, stderr=PIPE)

    cmd = ['awk','{print $3}']
    p[2] = Popen(cmd, stdin=p[2].stdout, stdout=PIPE, stderr=PIPE)

    stdout, stderr = p[2].communicate()
    if stdout:
        return stdout.strip()

    return 'unknown'

#----------------------------------------------------------------
#---- function
#----------------------------------------------------------------
def submit(cmdi,**kwargs):

    jid,ok,stop = _bsub(cmdi,**kwargs)

    stop_program(ok,stop)
    return jid

#----------------------------------------------------------------
#---- function
#----------------------------------------------------------------
def wait(JOBID=[],njob_target=0,closeout=False):
    ok,stop = _bwait(JOBID,njob_target)
    if not closeout:
        stop_program(ok,stop)
    return ok

#----------------------------------------------------------------
#---- function
#----------------------------------------------------------------
def status(jobid):
    return _bstat(jobid)

#----------------------------------------------------------------
#---- FUNCTION
#----------------------------------------------------------------
def stop_program(ok=False,stop=False):
    if not ok:
        if JID:
            print 'waiting on remaining jobs'
            ok = wait(closeout=True)
        print 'EXIT ERROR'
        sys.exit(1)
    elif stop:
        print('QUEUE TIMER EXPIRED')
        ok = wait(closeout=True)
        stop_program(ok)
        sys.exit(43)

#------------------------------------------------------------------------
#--- main
#------------------------------------------------------------------------
if __name__ == "__main__":
    '''
    call this script with task and arguments to use functions at command line
    tasks: submit, status, wait, peek
    example:
      wait on job(s)
        ./lsf_tools.py wait JOBIDs
    '''

    task = sys.argv[1]
    args = sys.argv[2:]

    if task == "submit":
        jid = submit(args)

    elif task == "status":
        stat = status(args[0])
        print statz

    elif task == "wait":
        wait(args)

    elif task == "peek":
        jout = glob(JOB_LOG_DIR+'/'+JOB_LOG_PREFIX+'.????????-??????.'+args[0]+'.out')
        jout.sort()
        if jout:
            try:
                call(['less',jout[-1]])
            except:
                call(['reset'])
                call(['clear'])
                exit()
        else:
            print args[0]+' not found.'

    else:
        print task+' not found.'
