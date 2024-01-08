import numpy as np
import random
from DeepESN import DeepESN
from utils import computeMusicAccuracy, config_pianomidi, load_pianomidi, select_indexes
class Struct(object): pass

# sistemare indici per IP in config_pianomidi, mettere da un'altra parte
# sistema selezione indici con transiente messi all'interno della rete
def main():
    
    # fix a seed for the reproducibility of results
    np.random.seed(7)
   
    # dataset path 
    path = 'datasets'
    dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes = load_pianomidi(path, computeMusicAccuracy)

    # load configuration for pianomidi task
    configs = config_pianomidi(list(TR_indexes) + list(VL_indexes))   
    
    # Be careful with memory usage
    Nr = 100 # number of recurrent units
    Nl = 5 # number of recurrent layers
    reg = 10.0**-2;
    transient = 5
    
    deepESN = DeepESN(Nu, Nr, Nl, configs)
    states = deepESN.computeState(dataset.inputs, deepESN.IPconf.DeepIP)
    
    train_states = select_indexes(states, list(TR_indexes) + list(VL_indexes), transient)
    train_targets = select_indexes(dataset.targets, list(TR_indexes) + list(VL_indexes), transient)
    test_states = select_indexes(states, TS_indexes)
    test_targets = select_indexes(dataset.targets, TS_indexes)
    
    deepESN.trainReadout(train_states, train_targets, reg)
    
    train_outputs = deepESN.computeOutput(train_states)
    train_error = error_function(train_outputs, train_targets)
    print('Training ACC: ', np.mean(train_error), '\n')
    
    test_outputs = deepESN.computeOutput(test_states)
    test_error = error_function(test_outputs, test_targets)
    print('Test ACC: ', np.mean(test_error), '\n')
 
 
if __name__ == "__main__":
    main()
    
