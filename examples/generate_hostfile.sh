# set hostfile_deepspeed & hostfile_mpich
echo "!!!please use generate_hostfile.sh before training"

# setting hostfile_mpich and hostfile_deepspeed
# this now supports setting up as many nodes as possible
# for examples:
#     1.$ bash generate_hostfile.sh #don't set hostfile
#     2.$ bash generate_hostfile.sh node1 #set one node
#     3.$ bash generate_hostfile.sh node1 node2 node3 node4 #set 4 nodes
#     4.$ bash generate_hostfile.sh node1 node2 node3 node4 node5 node6 node7 node8 #set 8 nodes
usage()
{
  echo "Example Usage:
            for 1 node: bash generate_hostfile.sh node1
            for 4 nodes: bash generate_hostfile.sh node1 node2 node3 node4"
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