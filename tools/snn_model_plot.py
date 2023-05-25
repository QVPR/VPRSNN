import matplotlib.pylab as plt 
import brian2 as b2



def plot_rateMonitors(rate_monitors, outputsPath, num_training_epochs, test_mode=False):

    plt.figure(figsize = (8, 8))
    for i, name in enumerate(rate_monitors):
        i += 1
        ax = plt.subplot(len(rate_monitors), 1, i, label=name)
        ax.plot(rate_monitors[name].t/b2.second, rate_monitors[name].rate, '.')
        ax.set(xlabel='Time (seconds)', ylabel='Rate (Hz)')
        ax.set_title('Rates of population ' + name)

        if not test_mode: 
            iter_len = int(max(rate_monitors[name].t/b2.second) / num_training_epochs)
            for i in range(num_training_epochs):
                if i % 2 != 0: 
                    continue 
                ax.axvspan(i*iter_len, i*iter_len+iter_len, facecolor='0.1', alpha=0.1)

    plt.tight_layout()
    plt.savefig(outputsPath + "rate monitors") 
    plt.close()


def plot_spikeMonitors(spike_monitors, outputsPath, num_training_epochs, test_mode=False):

    plt.figure(figsize = (8, 8))

    for i, name in enumerate(spike_monitors):
        i += 1
        ax = plt.subplot(len(spike_monitors), 1, i, label=name)
        ax.plot(spike_monitors[name].t/b2.second, spike_monitors[name].i, '.')
        ax.set(xlabel='Time (seconds)', ylabel='Neuron index')
        ax.set_title('Spikes of population ' + name)

        if not test_mode: 
            iter_len = int(max(spike_monitors[name].t/b2.second) / num_training_epochs)
            for i in range(num_training_epochs):
                if i % 2 != 0: 
                    continue 
                ax.axvspan(i*iter_len, i*iter_len+iter_len, facecolor='0.1', alpha=0.1)

    plt.tight_layout()    
    plt.savefig(outputsPath + "spike monitors")  
    plt.close()


def plot_spikeMonitorsCount(spike_monitors, outputsPath):

    plt.figure(figsize = (8, 8))
    
    for i, name in enumerate(spike_monitors):
        i += 1
        ax = plt.subplot(len(spike_monitors), 1, i, label=name)
        ax.plot(spike_monitors[name].count[:])
        ax.set(xlabel='Neuron index', ylabel='Number of spikes')
        ax.set_title('Spike count of population ' + name)
    
    plt.tight_layout()
    plt.savefig(outputsPath + "spike counters")  
    plt.close()



