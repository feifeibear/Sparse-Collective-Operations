ps aux | grep allreduce | awk '{print $2}' | xargs kill -9
