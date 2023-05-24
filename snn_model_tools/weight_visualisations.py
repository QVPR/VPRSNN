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




import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from snn_model import get_2d_input_weights


def plot_2d_input_weights(chosenCmap, weight_matrix, n_input, n_e, outputsPath):
    name = 'XeAe'
    weights = get_2d_input_weights(weight_matrix, n_input, n_e)
    fig = plt.figure(figsize = (18, 18))
    im2 = plt.imshow(weights, interpolation = "nearest", vmin = 0, cmap = chosenCmap) 
    plt.colorbar(im2)
    plt.title('weights of connection' + name)
    fig.canvas.draw()
    
    plt.savefig(outputsPath + str(fig.number))
    
    return im2, fig


def plot_raw_weights(values, outputsPath, name, chosenCmap):
    
    fig = plt.figure()
    img = plt.imshow(values, interpolation="nearest", cmap=chosenCmap, aspect='auto') 
    plt.colorbar(img)
    
    plt.xlabel('Target excitatory neuron number')
    plt.ylabel('Source excitatory neuron number')
    plt.title(name)
    plt.savefig(outputsPath + str(fig.number))


def plot_mean_XA_AA(XA_values, AA_values, n_e, outputsPath):
    
    XA_sum = np.nansum(XA_values, axis = 0)/n_e
    AA_sum = np.nansum(AA_values, axis = 0)/n_e

    fig = plt.figure(figsize=(22, 8))
    plt.plot(XA_sum, AA_sum, 'b.')
    for label, x, y in zip(range(len(XA_sum)), XA_sum, AA_sum):
        plt.annotate(label, xy=(x, y), xytext=(-0, 0),
                    textcoords = 'offset points', ha = 'right', va = 'bottom', color = 'k')
    plt.xlabel('summed input from X to A for A neurons')
    plt.ylabel('summed input from A to A for A neurons')
    plt.savefig(outputsPath + str(fig.number))    


def main():
    
    chosenCmap = cm.get_cmap('gray_r')

    weightsPath = './weights/weights_ne400_L100_offset0/'    
    outputsPath = './outputs/outputs_ne400_L100_offset0/'

    imWidth = 28
    imHeight = 28
    n_input = imWidth * imHeight
    n_e = 400

    readoutnames = ['XeAe', 'AeAi', 'AiAe' ] 

    for name in readoutnames:
        
        readout = np.load(weightsPath + name + '.npy')
        if (name == 'XeAe'):
            value_arr = np.nan * np.ones((n_input, n_e))
        else:
            value_arr = np.nan * np.ones((n_e, n_e))
        connection_parameters = readout

        for conn in connection_parameters: 

            src, tgt, value = int(conn[0]), int(conn[1]), conn[2]
            if np.isnan(value_arr[src, tgt]):
                value_arr[src, tgt] = value
            else:
                value_arr[src, tgt] += value
            values = np.asarray(value_arr)
        
        plot_raw_weights(values, outputsPath, name, chosenCmap)
        

        if name == 'XeAe': 
            XA_values = np.copy(values)
        if name == 'AeAi':
            AA_values = np.copy(values)

    plot_2d_input_weights(chosenCmap, XA_values, n_input, n_e, outputsPath)

    plot_mean_XA_AA(XA_values, AA_values, n_e, outputsPath)

    print('done')



if __name__ == "__main__":

    main()
    
    
