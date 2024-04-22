args=
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="${args} \"${i//\"/\\\"}\""
done

if [ "${args}" == "" ]; then args="/bin/bash"; fi

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

#TIP if use \ to expand commands into multiple lines, after the \, there can not be any more characters in that line. Otherwise, there will be an error.
singularity exec ${nv} \
    --overlay /scratch/fz2244/container/CDCL.ext3:rw \
    /scratch/work/public/singularity/cuda9.0-cudnn7-devel-ubuntu16.04-20201127.sif  \
    /bin/bash
