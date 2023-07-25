# set hostfile_deepspeed & hostfile_mpich
echo "!!!please use generate_hostfile.sh before training"

# use official mpich

# setting hostfile_mpich and hostfile_deepspeed
# this now supports setting up as many nodes as possible
# update for borealis
# for examples:
#     1.$ bash generate_hostfile.sh #don't set hostfile
#     2.$ bash generate_hostfile.sh x1001c2s0b0n0 #set one node
#     3.$ bash generate_hostfile.sh x1001c2s0b0n0 x1001c2s1b0n0 x1001c2s2b0n0 x1001c2s3b0n0 #set 4 nodes
#     4.$ bash generate_hostfile.sh x1001c2s0b0n0 x1001c2s1b0n0 x1001c2s2b0n0 x1001c2s3b0n0 x1001c2s4b0n0 x1001c2s5b0n0 x1001c2s6b0n0 x1001c2s7b0n0 #set 8 nodes
usage()
{
  echo "Example Usage:
            for 1 node: bash $0 x1001c2s0b0n0
            for 4 nodes: bash generate_hostfile.sh x1001c2s0b0n0 x1001c2s1b0n0 x1001c2s2b0n0 x1001c2s3b0n0"
  exit 2
}

if [ $# -gt 0 ]; then
    cat /dev/null > $LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_mpich
    cat /dev/null > $LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_deepspeed
    mid=" slots="
    slots=12
    for i in "$@"; do
        host=$i
        host_slot="$i$mid$slots"
        echo $host>>$LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_mpich
        echo $host_slot>>$LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_deepspeed
    done
else
    usage
fi