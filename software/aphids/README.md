# APHIDS #

APHIDS stands for: **A**daptive **P**hased-array and **H**eterogenous **I**nterpolator and **D**ownsampler for **S**WARM.

## Caveats for running APHIDS ##

In order to run APHIDS as a regular user with large shared memory buffers you will need to create the following file:

```
# /usr/security/limits.d/memlock.conf
# Set the memlock limits higher

*               soft    memlock            64
*               hard    memlock            1024
```

Where the 1024 (which is in KB) can be as big as you are comfortable making it. Then you will need to bring the soft 
limit up as a regular user using the following command:

```
$ ulimit -S -l 1024
```

Alternatively you can just bring both the soft and hard limits up to whatever value you want.

Additionally, you will need to add the following line to /etc/sysctl.conf:

```
# Controls the maximum number of semaphore arrays
kernel.sem = 1024 256000 32 4096
```

Where the first number is the maximum number of allowed semaphores. For APHIDS/hashpipe this should match the buffer size 
you are trying to achieve. You must then reload the sysctl (as root) configuration with:

```
$ sysctl -p
```
