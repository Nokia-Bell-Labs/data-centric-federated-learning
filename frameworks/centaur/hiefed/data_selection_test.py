import numpy as np
from data_selection import CDFSelection

# Test on data selection module

test_size = 10
losses = np.random.uniform(low=0, high=2, size=(test_size,))

skip_steps = test_size - 1
alpha = 5
beta = 2
gamma = 1

# select data based on loss with build_queue
queue_size = int(test_size/2)
loss_selec1 = CDFSelection(alpha, beta, max_len=queue_size)
step = 1
for loss in losses:
    loss_selec1.maintain_queue(loss)
    if step < skip_steps:
        step += 1
    else:
        prob_drop, prob_ap = loss_selec1.query_prob(loss)

print("=======================")

# select data based on loss with full_queue
loss_selec2 = CDFSelection(alpha, beta, full_queue=losses)
idx = 0
for loss in losses:
    prob_drop, prob_ap = loss_selec2.query_prob(loss)
    #loss_selec2.update_state(0.5, idx)
    idx += 1
