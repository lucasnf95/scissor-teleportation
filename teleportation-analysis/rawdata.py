
import numpy as np
from scipy.fft import fft, fftfreq
import os
from qopt import lecroy
import matplotlib.pyplot as plt
import functions
import inputstate

class rawdata:
    def __init__(self, path, dimension, tomography_phases = ['000', '030', '060', '090', '120', '150'], phase_vacuum = True):
        self.path = path
        self.dimension = dimension
        self.tomography_phases = tomography_phases
        self.phase_vacuum = phase_vacuum
        self.homodyne, self.heralding, self.charlie, self.vacuum, self.meta = self.import_data()

    def import_data(self):
        '''
        Import teleportation data
        Parameters
            path: string
                folder on which data is saved
            phase_vacuum: boolean
                True if each vacuum measurement is associated with a different phase angle
        Returns

        '''
        homodyne_files = {self.tomography_phases[i]: \
                          np.array([os.path.join(self.path, f) for f in sorted(os.listdir(self.path)) if "C1tele" + self.tomography_phases[i] in f]) \
                          for i in range(len(self.tomography_phases))}
        heralding_files = {self.tomography_phases[i]: \
                          np.array([os.path.join(self.path, f) for f in sorted(os.listdir(self.path)) if "C2tele" + self.tomography_phases[i] in f]) \
                          for i in range(len(self.tomography_phases))}
        charlie_files = {self.tomography_phases[i]: \
                          np.array([os.path.join(self.path, f) for f in sorted(os.listdir(self.path)) if "C3tele" + self.tomography_phases[i] in f]) \
                          for i in range(len(self.tomography_phases))}
        if self.phase_vacuum:
            vacuum_files = {self.tomography_phases[i]: \
                              np.array([os.path.join(self.path, f) for f in sorted(os.listdir(self.path)) if "C1vac" + self.tomography_phases[i] in f]) \
                              for i in range(len(self.tomography_phases))}
        else:
            vacuum_files = {self.tomography_phases[i]: \
                              np.array([os.path.join(self.path, f) for f in sorted(os.listdir(self.path)) if "C1vac" in f]) \
                              for i in range(len(self.tomography_phases))}

        self.files_for_each_phase = len(homodyne_files[self.tomography_phases[0]])

        homodyne = {phase: lecroy.read(homodyne_files[phase][0])[2] for phase in self.tomography_phases}
        heralding = {phase: lecroy.read(heralding_files[phase][0])[2] for phase in self.tomography_phases}
        charlie = {phase: lecroy.read(charlie_files[phase][0])[2] for phase in self.tomography_phases}
        vacuum = {phase: lecroy.read(vacuum_files[phase][0])[2] for phase in self.tomography_phases}
        if self.files_for_each_phase > 1:
            for _, phase in enumerate(self.tomography_phases):
                for j in range(self.files_for_each_phase-1):
                    homodyne[phase] = np.concatenate((homodyne[phase], lecroy.read(homodyne_files[phase][j+1])[2]))
                    heralding[phase] = np.concatenate((heralding[phase], lecroy.read(heralding_files[phase][j+1])[2]))
                    charlie[phase] = np.concatenate((charlie[phase], lecroy.read(charlie_files[phase][j+1])[2]))
                if len(vacuum_files[phase]) > 1:
                    vacuum[phase] = np.concatenate((vacuum[phase], lecroy.read(vacuum_files[phase][0])[2]))
        
        # Extract information from metadata
        meta = lecroy.read(heralding_files[self.tomography_phases[0]][0])[0]
        self.sequences = meta['subarray_count']
        self.total_sequences = self.files_for_each_phase*self.sequences
        self.total_points = meta['wave_array_count']
        self.points_per_seq = int(meta['wave_array_count']/self.sequences)
        self.Ts = meta['horiz_interval']
        
        # Electronic noise
        elec_file = np.array([os.path.join(self.path, f) for f in sorted(os.listdir(self.path)) if "C1elec" in f])
        elec = lecroy.read(elec_file[0])[2]

        print('Data imported')
        self.check_clearance(homodyne, vacuum, elec)

        return homodyne, heralding, charlie, vacuum, meta

    def check_clearance(self, homodyne, vacuum, elec):
        tr000 = homodyne[self.tomography_phases[0]]
        trvac = vacuum[self.tomography_phases[0]]
        trelec = elec
        freq = fftfreq(502, .002)

        plt.figure(figsize=(10,6))
        plt.plot(freq[:60], 10*np.log10((np.abs(fft(trvac))**2).mean(0))[:60], color='orange', label='Vacuum')
        plt.plot(freq[:60], 10*np.log10((np.abs(fft(tr000))**2).mean(0))[:60], color='green', label='Signal')
        plt.plot(freq[:60], 10*np.log10((np.abs(fft(trelec))**2).mean(0))[:60], color='blue', label='Electronic')
        plt.legend()
        plt.grid()
        plt.xlabel('MHz')
        plt.ylabel('Noise power [dB]')

        # Calculate clearance for a desired frequency
        freq_clear = 10
        min_f = [np.abs(f - 10) for f in freq[:60]]
        min_p = np.where(min_f == np.min(min_f))[0][0]

        vac_clear = 10*np.log10((np.abs(fft(trvac))**2).mean(0))[:60][min_p]
        elec_clear = 10*np.log10((np.abs(fft(trelec))**2).mean(0))[:60][min_p]

        clearance = vac_clear - elec_clear
        print('Clearance at %.1f MHz is %.2f dB' %(freq_clear, clearance))

    def fluctuations_conference(self, conf_homodyne, conf_vacuum, mf):
        '''
        Check fluctuations suffered by data during measurements
        Parameters
            homodyne: dict
                raw measurement data
            vacuum: dict
                raw vacuum data
        '''
        # Conference performed for raw data with mode function applied
        conf_vacuum = {phase: np.zeros((self.sequences)) for phase in self.tomography_phases}
        conf_homodyne = {phase: np.zeros((self.total_sequences)) for phase in self.tomography_phases}

        for i, phase in enumerate(self.tomography_phases):
            for k in range(self.total_sequences):
                conf_homodyne[phase][k] = np.dot(homodyne[phase][k], self.mf)
                if k < sequences:
                    conf_vacuum[phase][k] = np.dot(vacuum[phase][k], self.mf)

        print('CONFERENCE OF FLUCTUATIONS OF OUTPUT DATA')
        #Homodyne data
        w = 200
        fig, axs = plt.subplots(3,2, figsize=(15,12), sharey=True, sharex=True)
        for i, ax in enumerate(axs.flat):
            ax.plot(conf_homodyne[self.tomography_phases[i]], '.', alpha=.1)
            ax.plot(np.convolve(conf_homodyne[self.tomography_phases[i]], np.ones(w), 'same') / w)
            div = make_axes_locatable(ax)
            axB = div.append_axes("right", 1, pad=0.1, sharey=ax)
            axB.xaxis.set_tick_params(labelbottom=False)
            axB.yaxis.set_tick_params(labelleft=False)
            axB.text(.02, 3.5, 'var.: {:.2f}'.format(conf_homodyne[self.tomography_phases[i]].var()))
            bins = np.linspace(-4,4,101)
            axB.hist(conf_homodyne[self.tomography_phases[i]], bins=bins, density=True, orientation='horizontal')
            plt.title('Homodyne measurement phase '+ self.tomography_phases[i], loc = 'center')

        #Vacuum data
        fig, axs = plt.subplots(3,2, figsize=(15,12), sharey=True, sharex=True)
        for i, ax in enumerate(axs.flat):
            ax.plot(conf_vacuum[self.tomography_phases[i]], '.', alpha=.1)
            ax.plot(np.convolve(conf_vacuum[self.tomography_phases[i]], np.ones(w), 'same') / w)
            div = make_axes_locatable(ax)
            axB = div.append_axes("right", 1, pad=0.1, sharey=ax)
            axB.xaxis.set_tick_params(labelbottom=False)
            axB.yaxis.set_tick_params(labelleft=False)
            axB.text(.02, 3.5, 'var.: {:.2f}'.format(conf_vacuum[self.tomography_phases[i]].var()))
            bins = np.linspace(-4,4,101)
            axB.hist(conf_vacuum[self.tomography_phases[i]], bins=bins, density=True, orientation='horizontal')
            plt.title('Vacuum measurement phase '+ self.tomography_phases[i], loc = 'center')

    def offset_correct(self, file, block_size = 100):
        '''
        Correct offset by considering last few points as vacuum.
        Parameters:
            file: dictionary
                data organized by phase
            block_size: int
                amount of points for which data will be divided
        Return:
            file_corr: corrected data
        '''
        seq = len(file['000'])
        file_corr = {}
        block_size = 100
        for k,v in file.items():
            tails = v[:,-(self.points_per_seq//5):].reshape((seq//block_size, block_size, -1))
            tail_avg = tails.mean((1,2))
            _file_corr = v.reshape((seq//block_size, block_size, -1)) - tail_avg[:, np.newaxis, np.newaxis]
            _file_corr = _file_corr.reshape((seq, self.points_per_seq))
            file_corr[k] = _file_corr
        return file_corr

    def plot_mode_function(self, file, homodyne_g = 2*np.pi*17e6, homodyne_k = 2*np.pi*30e6, homodyne_time_delay = -250e-9, correct = False):
        '''
        Plot mode function along with variance across each data point to check fitting.
        Parameters:
            file: dictionary
                data organized by phase
        Return:
            mf: 1D array
                mode function
        '''
        # Correct data offset
        if correct:
            file_corr = self.offset_correct(file)
        else:
            file_corr = file

        # Mode function with time information from data
        begin = self.meta['horiz_offset']
        x = np.linspace(begin, begin + self.Ts*self.total_points, self.total_points)
        x_sequence = np.linspace(begin, begin + self.Ts*self.points_per_seq, self.points_per_seq)
        self.mf = functions.mode_function(homodyne_k, homodyne_g, homodyne_time_delay, x_sequence)

        seq_var = np.array([np.var(file_corr[phase], axis=0) for phase in self.tomography_phases])
        mean_seq_var = np.mean(seq_var)
        mf_norm = (self.mf - self.mf.mean())*(seq_var.mean(axis=0).max().max() - mean_seq_var)/(self.mf.max() - self.mf.mean() + mean_seq_var)

        # Plot variance of data and normalized mode function
        print('VARIANCE ACROSS EACH POINT OF TELEPORTED DATA')
        plt.figure()
        plt.plot(x_sequence, seq_var.mean(axis=0) - mean_seq_var)
        plt.legend()
        plt.plot(x_sequence, mf_norm)
        plt.title('Variance of measured signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Variance (V²)')
        plt.show()

        return self.mf

    def apply_mode_function(self, homodyne, vacuum, homodyne_g = 2*np.pi*17e6, homodyne_k = 2*np.pi*30e6, homodyne_time_delay = -250e-9, input_folder = None, correct = True, check_fluctuations = False, input_plot = True):
        '''
        Apply temporal mode function to the data.
        Data is corrected and normalized to vacuum shot noise (1/2).
        Parameters
            input_folder: str
                if not None, the input state is also analysed
            homodyne: dictionary
                homodyne measurements
            vacuum: dictionary
                vacuum measurements
        Return:
            mf_homodyne: dictionary
                homodyne measurements with mode function applied
            mf_vacuum: dictionary
                vacuum measurements with mode function applied
        '''
        if correct:
            H_corr = self.offset_correct(homodyne)
            V_corr = self.offset_correct(vacuum)
        else:
            H_corr = homodyne
            V_corr = vacuum

        self.mf = self.plot_mode_function(homodyne, homodyne_g = homodyne_g, homodyne_k = homodyne_k, homodyne_time_delay = homodyne_time_delay)

        if check_fluctuations:
            self.fluctuations_conference(homodyne, vacuum)

        # Apply mode function to data
        mf_vacuum = {phase: np.zeros((self.sequences)) for phase in self.tomography_phases}
        mf_homodyne = {phase: np.zeros((self.total_sequences)) for phase in self.tomography_phases}
        seq_var = {phase: np.zeros((self.total_sequences)) for phase in self.tomography_phases}

        for i, phase in enumerate(self.tomography_phases):
            for k in range(self.total_sequences):
                mf_homodyne[phase][k] = np.dot(H_corr[phase][k], self.mf)
                if k < self.sequences:
                    mf_vacuum[phase][k] = np.dot(V_corr[phase][k], self.mf)
            seq_var[phase] = np.var(H_corr[phase], axis = 0)

        # Normalize data to vacuum level
        mean_vac = {phase: np.mean(mf_vacuum[phase]) for phase in self.tomography_phases}
        var_vac = {phase: np.var(mf_vacuum[phase]) for phase in self.tomography_phases}

        for i, phase in enumerate(self.tomography_phases):
            mf_homodyne[phase] = (mf_homodyne[phase] - mean_vac[phase])/np.sqrt(2*var_vac[phase])
            mf_vacuum[phase] = (mf_vacuum[phase] - mean_vac[phase])/np.sqrt(2*var_vac[phase])

        if input_folder == None:
            return mf_homodyne, mf_vacuum
        else:
            input_rho = inputstate.reconstruct_input_state(input_folder, self.mf, input_plot = input_plot)
            return mf_homodyne, mf_vacuum, input_rho
