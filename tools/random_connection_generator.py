'''
MIT License

Copyright (c) 2015 Peter U. Diehl and Matthew Cook

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''




import argparse
from pathlib import Path

import numpy as np




def main(args):

    np.random.seed(0)
    
    dataPath = './random/random_ne{}{}/'.format(args.n_e, args.ad_path)
    Path(dataPath).mkdir(parents=True, exist_ok=True)

    imWidth = 28 
    imHeight = 28
    n_e = args.n_e
    n_i = args.n_e

    create_weights(dataPath, imWidth, imHeight, n_e, n_i)
    
    print('done')
    
    


def sparsenMatrix(baseMatrix, pConn):
    
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0]*int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
            
    return weightMatrix, weightList
        
    
def create_weights(dataPath, imWidth=28, imHeight=28, n_e=400, n_i=400):
    
    n_input = imWidth * imHeight   

    weight = {}
    weight['ee_input'] = 0.3 
    weight['ei'] = 10.4
    weight['ie'] = 17.0
    pConn = {}
    pConn['ee_input'] = 1.0 
    pConn['ei'] = 0.0025
    pConn['ie'] = 0.9
    
        
    print('create random connection matrices')
    connNameXeAe = 'XeAe'
    weightMatrix = np.random.random((n_input, n_e)) + 0.01
    weightMatrix *= weight['ee_input']
    
    if pConn['ee_input'] < 1.0:
        weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
    else:
        weightList = [(i, j, weightMatrix[i,j]) for j in range(n_e) for i in range(n_input)]
    np.save(dataPath+connNameXeAe, weightList)
        
    
    print('create connection matrices from E->I which are purely random')
    connNameAeAi = 'AeAi'
    if n_e == n_i:
        weightList = [(i, i, weight['ei']) for i in range(n_e)]
    else:
        weightMatrix = np.random.random((n_e, n_i))
        weightMatrix *= weight['ei']
        weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
    print('save connection matrix', connNameAeAi)
    np.save(dataPath+connNameAeAi, weightList)
                
        
    print('create connection matrices from I->E which are purely random')
    connNameAiAe = 'AiAe'
    if n_e == n_i:
        weightMatrix = np.full((n_i, n_e), weight['ie'])
        np.fill_diagonal(weightMatrix, 0)
        weightList = [(i, j, weightMatrix[i,j]) for i in range(n_i) for j in range(n_e)]
    else:
        weightMatrix = np.random.random((n_i, n_e))
        weightMatrix *= weight['ie']
        weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
    print('save connection matrix', connNameAiAe)
    np.save(dataPath+connNameAiAe, weightList)


    
    

if __name__ == "__main__":
    
    n_e = 400 
    seed = 0
    ad_path = ""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_e', type=int, default=n_e)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--ad_path', type=str, default=ad_path)

    parser.set_defaults()
    args = parser.parse_args()
    print(args)

    main(args)

    










