import os
from subprocess import call,Popen,PIPE
import tempfile

#--------------------------------------------------------
#--- FUNCTION
#--------------------------------------------------------
def ncl_vardef(kwargs):
    cmd = []
    for k,v in kwargs.items():
        if isinstance(v, (str)):
            cmd.append('%s="%s"'%(k,v))
        elif isinstance(v, (int,long)):
            cmd.append('%s=%d'%(k,v))
        elif isinstance(v, (float)):
            cmd.append('%s=%e'%(k,v))
        else:
            print('type: of var: '+k+' not supported')
            exit(1)
    return cmd

#--------------------------------------------------------
#-- interface to NCL
#--------------------------------------------------------
def ncl(script_ncl,var_dict):

    #-- make temporary "exit" file (NCL must delete)
    (f,exitfile) = tempfile.mkstemp('','ncl-abnormal-exit.')
    var_dict.update({'ABNORMAL_EXIT': exitfile})
    
    #-- call NCL
    cmd = ['ncl','-n','-Q']+ncl_vardef(var_dict)+[script_ncl]
    p = Popen(cmd,stdout=PIPE,stderr=PIPE)
    out,err = p.communicate()

    #-- error?  if return code not zero or exitfile exists
    ok = (p.returncode == 0) and (not os.path.exists(exitfile))
    
    #-- print stdout and stderr
    if out: print(out)
    if err: print(err)

    if not ok:
        print('NCL ERROR')
        if os.path.exists(exitfile): call(['rm','-f',exitfile])
        return False
    
    return True    
