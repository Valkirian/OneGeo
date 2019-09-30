
#!/bin/bash

SESS="linked"
stage_dir="$HOME/microscope"
main(){

    tmux new -s $SESS -d

    tmux send -t "$SESS:1" Enter "sudo sshfs -o idmap=user ops@desarrollo7:/home/ops/code ${stage_dir}" Enter
    sudo dolphin ${stage_dir}

}
