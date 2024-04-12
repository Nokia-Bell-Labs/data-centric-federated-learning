# Centaur: Enhancing Efficiency in Multidevice Federated Learning through Data Selection

This is one implementation of a partitioned-based federated learning (FL) framework to support efficient FL training on constrained edge devices.


## Structure
```bash
├── running_args.py     // arguments and settings
├── main.py             // main entry - server side
├── oam.py              // mobility model matrix
├── hiefed                      // main functions
│   ├── client.py               // client side
│   ├── dataset_utils.py        // dataset manipulation
│   ├── data_selection.py       // data selection module
│   ├── data_selection_test.py  // tests of the data selection module
│   └── fed_criterion.py        // customized Flower criterion
├── measures                 // measure-related functions
│   ├── mac_comm_counter.py  // counters of computation and communication
│   └── coverage.py          // online and offline of devices 
├── plot                // result plotting
│   ├── plot_utils.py   // plot util functions
│   └── 0-9_*.py        // plots of 0 to 9
├── third_party                     // third party functions
│   ├── dataset_partition_flwr.py   // dataset partition of Flower simulation
│   ├── autograd_hacks.py           // fast autograd functions
│   └── autograd_hacks_test.py      // tests of autograd functions
├── make_script.py              // make scripts to replicate results
└── client_configurations.csv   // configurations of client devices 
```


## Prerequisites
First, install core libraries:
```
pip install flwr
pip install -U ray==1.11.1
```
Tested on Python 3.7.10 and Torch 1.12.1. 

Note that, the current Flower version (1.0.0) has a problem when running a simulation with ray>=1.12. Ray workers will go into `Ray::IDLE` mode, which occupies CUDA memory and leads to OOM. **For using ray>=1.12 only**, a workaround is to change all `ray.remote` as `ray.remote(max_calls=1)` in the Flower's [`ray_client_proxy`](https://github.com/adap/flower/blob/main/src/py/flwr/simulation/ray_transport/ray_client_proxy.py) file.



## Test
Create folders for logging and saving evaluation results:
```
mkdir log; mkdir results
```

Then, try to run Centaur with default arguments with:
```
python main.py
```
Note that you may need to add this directory to Python path with `export PYTHONPATH=$PYTHONPATH:path/to/this/dir`.

For replicating results, run the following command to generate bash scripts.
```
python make_script.py
```
and then run specific `.sh` file for evaluations.
