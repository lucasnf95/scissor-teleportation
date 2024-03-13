
import numpy as np
import matplotlib.pyplot as plt

class selectdata:
    def __init__(self, mf_homodyne, charlie, meta, time_window, time_delay = 95e-9, peak_limit = -.5, tomography_phases = ['000', '030', '060', '090', '120', '150'], verbose = True):
        self.mf_homodyne = mf_homodyne
        self.charlie = charlie
        self.meta = meta
        
        self.peak_limit = peak_limit
        self.time_window = time_window
        self.time_delay = time_delay
        self.tomography_phases = tomography_phases
        self.verbose = verbose
        
        self.sequences = self.meta['subarray_count']
        self.total_sequences = len(charlie[self.tomography_phases[0]])
        self.files_for_each_phase = int(self.total_sequences/self.sequences)
        self.total_points = self.meta['wave_array_count']
        self.points_per_seq = int(self.meta['wave_array_count']/self.sequences)
        self.Ts = self.meta['horiz_interval']
        
    def select_teleported_data(self):
        sequence_with_coincidence, success_rate = self.calculate_success_rate()
        sig = {phase: np.zeros(len(sequence_with_coincidence[phase])) for phase in self.tomography_phases} # Homodyne signal for the data where coincidence was measured
        for phase in self.tomography_phases:
            for s, seq in enumerate(sequence_with_coincidence[phase]):
                sig[phase][s] = self.mf_homodyne[phase][seq]

        return sig, success_rate

    def calculate_success_rate(self):
        coincidence_peaks = {phase: [] for phase in self.tomography_phases}
        sequence_with_coincidence = {phase: [] for phase in self.tomography_phases}
        success_rate = []
        heralding_peak_position = int(self.points_per_seq/2) # The heralding peak is the middle points of the sequence
        # Only search for Charlie peaks in the vicinity defined previously
        first_charlie_point = int(heralding_peak_position + (self.time_delay - self.time_window)/self.Ts)
        last_charlie_point = int(heralding_peak_position + (self.time_delay + self.time_window)/self.Ts) + 1

        for i, phase in enumerate(self.tomography_phases):
        #     peak_limit = np.max(charlie[phase]) - 0.5
            for k in range(len(self.charlie[self.tomography_phases[0]])):
        #        possible_peak = peakdetect(y_axis = np.diff(charlie[phase][k])[first_charlie_point:last_charlie_point], \
        #                                    x_axis = x_sequence[first_charlie_point:last_charlie_point], lookahead = 1)[1]
                d = np.diff(self.charlie[phase][k])[first_charlie_point:last_charlie_point]
                possible_peak = np.argmin(d)
                if d[possible_peak] < self.peak_limit:
                    coincidence_peaks[phase].append(possible_peak)
                    sequence_with_coincidence[phase].append(k)
        #        for l in range(len(possible_peak)):
        #            if possible_peak[l][1] < peak_limit:
        #                coincidence_peaks[phase].append(possible_peak[l])
        #                sequence_with_coincidence[phase].append(k)
            amount = len(coincidence_peaks[phase])
            success_rate.append(amount/len(self.charlie[self.tomography_phases[0]]))
            if self.verbose:
                print('Amount of coincidence clicks for phase %s: %d out of %d' %(phase, amount, self.files_for_each_phase*self.sequences))

        total_success_rate = np.sum(success_rate)/len(self.tomography_phases)
        percentage_of_success = 100*total_success_rate
        print('Success rate = %.2f %%' %percentage_of_success)

        return sequence_with_coincidence, total_success_rate

    def teleported_state_with_optimal_success_rate(self):    

        sequence_with_coincidence, success_rate = self.find_optimal_success_rate()
        sig = {phase: np.zeros(len(sequence_with_coincidence[phase])) for phase in self.tomography_phases} # Homodyne signal for the data where coincidence was measured
        for phase in self.tomography_phases:
            for s, seq in enumerate(sequence_with_coincidence[phase]):
                sig[phase][s] = self.mf_homodyne[phase][seq]

        return sig, success_rate

    def find_optimal_success_rate(self, extra_delay = 5e-9, n_delays = 50):

        # Check which delay yields the maximum amount of coincidences
        heralding_peak_position = int(points_per_seq/2)
        delays = []
        for i in range(n_delays):
            if int(heralding_peak_position + (self.time_delay + extra_delay*(i-n_delays/2) + self.time_window)/Ts) < self.points_per_seq:
                delays.append(extra_delay*(i-n_delays/2))
        delays_str = [str(d) for d in delays]
        dict_sequence_with_coincidence = {d: None for d in delays_str}
        dict_success_rate = {d: None for d in delays_str}
        for d, d_str in enumerate(delays_str):
            if d % 10 == 0:
                print(d)      
            test_delay = self.time_delay + delays[d]
            dict_sequence_with_coincidence[d_str], dict_success_rate[d_str] = self.calculate_success_rate()

        maximum_success_key = max(dict_success_rate, key = dict_success_rate.get)
        #maximum_success_point = np.where(delays_str == maximum_success_key)[0]
        maximum_success_rate = dict_success_rate[maximum_success_key]
        print('Maximum %.2f%% success rate for delay %.0f ns'%(100*maximum_success_rate, float(maximum_success_key)*1e9))

        max_success_rate_list = [dict_success_rate[d] for d in delays_str]
        plt.figure(figsize = [10,8])
        #plt.plot(delays, 100*mean_success)
        plt.plot(delays, 100*np.array(max_success_rate_list))
        plt.xlabel('Extra time delay')
        plt.ylabel('Success rate (%)')
        plt.title('Success rate as a function of time delay (s)')

        return dict_sequence_with_coincidence[maximum_success_key], maximum_success_rate
