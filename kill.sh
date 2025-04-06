HOSTFILE=/home/zhouenshen/code/VILA/hostfile/hostfile_ours
NNodes=`cat ${HOSTFILE} | wc -l`
i=1
for ip in `cat ${HOSTFILE} | cut -d " " -f1`
do
    echo "Starting node ${i}/${NNodes}: ${ip}"
    ssh $ip \
    "pkill -f torchrun && pkill -f deepspeed" 
    i=`expr $i + 1`
done