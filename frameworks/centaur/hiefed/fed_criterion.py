import flwr as fl
from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
)
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.criterion import Criterion


class FedAvg_criterion(fl.server.strategy.FedAvg):

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        #_# Return the sample size and the required number of available clients.
        #_# https://flower.dev/docs/apiref-flwr.html#flwr.server.strategy.FedAvg.num_fit_clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        
        # load cids from the buffer
        with open('log/cids_buffer.csv', 'r') as f:
            pre_cids = f.read()

        #_# https://github.com/adap/flower/blob/main/src/py/flwr/server/criterion.py
        #_# Abstract class which allows subclasses to implement criterion sampling.
        class ChosenCriterion(Criterion):
            """Criterion to select only test clients."""
            #_# Decide whether a client should be eligible for sampling or not.
            #_# pre_cids means previous client ids.
            #_# for example: 47|170,94|170,63|170,58|170,71|170,68|170,64|170,5|170,
            
            def select(self, client: ClientProxy) -> bool:
                if pre_cids == '':
                    return True
                else:
                    cids_only = [i.split('|')[0] for i in pre_cids.split(',')[:-1]]
                    return client.cid in cids_only
        #_#
        """
        Current code is like this: we use the same "Criterion" for both "iot" and "ap" rounds.
        Because for "iot" rounds we do not need to choose specific clients, after each
        "ap" round, everytime we restart the cids buffer in "client.py":
        if destine == 'ap': 
            with open('log/cids_buffer.csv', 'w') as f:
                f.write('')
        So, for "iot" round, in the above "select()" method the "if" condition is true.
        Maybe, it would be better to implement two "Criterion"s: one dedicated to "iot" and another to "ap".
        """
        #_#

        clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=min_num_clients,
            criterion=ChosenCriterion()
        )
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]