# set hostfile_deepspeed & hostfile_mpich
echo "!!!please use generate_hostfile.sh before training"

# use official mpich

# setting hostfile_mpich and hostfile_deepspeed
# this now supports setting up as many nodes as possible
# update for borealis
# for examples:
#     1.$ bash generate_hostfile.sh #don't set hostfile
#     2.$ bash generate_hostfile.sh x10001 #set one node
#     3.$ bash generate_hostfile.sh x10001 x10002 x10003 x10004 #set 4 nodes
#     4.$ bash generate_hostfile.sh x10001 x10002 x10003 x10004 x10005 x10006 x10007 x10008 #set 8 nodes
# update for OAM system
# for examples:
#     1.$ bash generate_hostfile.sh #don't set hostfile
#     2.$ bash generate_hostfile.sh oam compute1 #set one compute node
#     3.$ bash generate_hostfile.sh oam compute1 compute2 #set 2 compute nodes
usage()
{
  echo "Example Usage:
            for 1 node: bash $0 x10001
            for 4 nodes: bash generate_hostfile.sh x10001 x10002 x10003 x10004"
  exit 2
}

if [ $# -gt 0 ]; then
    cat /dev/null > $LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_mpich
    cat /dev/null > $LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_deepspeed
    mid=" slots="
    slots=12
    for i in "$@"; do
        if [ "$i" == oam ]; then
            slots=8
        else
            host=$i
            host_slot="$i$mid$slots"
            echo $host>>$LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_mpich
            echo $host_slot>>$LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_deepspeed
        fi
    done
else
    usage
fi
