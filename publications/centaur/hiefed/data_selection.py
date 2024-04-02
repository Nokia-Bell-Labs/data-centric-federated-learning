# data selection module
class CDFSelection:
    def __init__(self, alpha, beta, max_len=None, full_queue=None):
        self.alpha = alpha
        self.beta = beta
        # for loss, both alpha and beta should be larger than 1, in order to 
        # provide the middle interval for norm measurement later.

        if full_queue is not None:
            self.queue = full_queue
            self.max_len = len(full_queue)
        else:
            self.queue = []
            self.max_len = max_len

    # build the queue and keep the size with poping out old instance.
    def maintain_queue(self, loss):
        if len(self.queue) == self.max_len:
            self.queue.pop(0)
        self.queue.append(loss)

    # update state of the queue
    def update_state(self, loss, index):
        self.queue[index] = loss

    # query the probably to drop or go ap
    def query_prob(self, loss):
        index = sorted(self.queue).index(loss) + 1

        cdf_left = float(index)/self.max_len
        cdf_right = 1 - cdf_left

        # only chose when self.alpha != 0
        if self.alpha == 0:
            prob_drop = 0
        else:
            prob_drop = cdf_right**self.alpha
        
        # only chose when self.beta != 0
        if self.beta == 0:
            prob_ap = 0
        else:
            prob_ap = cdf_left**self.beta

        # print(self.queue)
        # print("cdf_left={}, prob_low(drop)={}".format(cdf_left, prob_drop))
        # print("cdf_right={}, prob_high(ap)={}".format(cdf_right, prob_ap))
        # print("-----")

        return prob_drop, prob_ap


# TODO: data selection based on the built distribution