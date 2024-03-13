
import numpy as np
import os
import matplotlib.pyplot as plt
from qopt import lecroy
import functions

class inputstate:
    
    def __init__(self, input_folder, dimension, modefunction, input_plot = True):
        self.input_folder = input_folder
        self.dimension = dimension
        self.mf = modefunction
        self.input_plot = input_plot
        self.input_rho = self.reconstruct_input_state()

    def import_input_files(self):
        '''
        Import files for input state analysis
        Parameters:
            input_folder: str
                Folder in which data is saved
        Return:
            input_data: dict
                Input files separated by measured phase
            samplehold: dict
                Samples hold files separared by measured phase
            input_meta: list
                Metadata
        '''
        # Import files
        input_files_000 = np.array([os.path.join(self.input_folder, f) for f in sorted(os.listdir(self.input_folder)) if "C1input000" in f])
        input_files_090 = np.array([os.path.join(self.input_folder, f) for f in sorted(os.listdir(self.input_folder)) if "C1input090" in f])
        samplehold_files_000 = np.array([os.path.join(self.input_folder, f) for f in sorted(os.listdir(self.input_folder)) if "C3input000" in f])
        samplehold_files_090 = np.array([os.path.join(self.input_folder, f) for f in sorted(os.listdir(self.input_folder)) if "C3input090" in f])

        # Read first file
        input_000 = lecroy.read(input_files_000[0])[2]
        input_090 = lecroy.read(input_files_090[0])[2]
        samplehold_000 = lecroy.read(samplehold_files_000[0])[2]
        samplehold_090 = lecroy.read(samplehold_files_090[0])[2]

        # Read rest of the files
        if len(input_files_000) > 1:
            for i in range(len(input_files_000)-1):
                input_000 = np.concatenate(input_000, lecroy.read(input_files_000[i+1])[2])
                input_090 = np.concatenate(input_090, lecroy.read(input_files_090[i+1])[2])
                samplehold_000 = np.concatenate(samplehold_000, lecroy.read(samplehold_files_000[i+1])[2])
                samplehold_090 = np.concatenate(samplehold_090, lecroy.read(samplehold_files_090[i+1])[2])

        # Create dictionary
        input_phases = ['000', '090']
        input_data = {'000': input_000, '090': input_090}
        samplehold = {'000': samplehold_000, '090': samplehold_090}
        input_meta = lecroy.read(input_files_000[0])[0]

        return input_data, samplehold, input_meta

    def read_folder(self):
        '''
        Take input state parameters from folder name.
        Parameters:
            input_folder: str
                Folder where input files (or teleported files) are located.
        Return:
            input_alpha: float
                Amplitude of input state
            input_theta: float
                Phase (degrees) of input state
        '''
        #begin_theta = input_folder.find('phase')+len('phase')
        #rest_of_folder = input_folder[begin_theta:]
        #end_theta = begin_theta + rest_of_folder.find('/')
        relay_theta = 90#int(input_folder[begin_theta:end_theta]) # Degrees
        if 'size' in self.input_folder:
            # Get input state values from folder name
            text_size = self.input_folder[self.input_folder.find('size') + len('size'): self.input_folder.find('-')]
            input_angle = 0
            if text_size == '0':
                input_alpha = 0
            else:
                input_alpha = int(text_size[text_size.find('0')+1:])/(10**(len(text_size)-1))
        else:
            input_alpha, input_angle = self.calculate_input_parameters()
        return input_alpha, relay_theta, input_angle

    def calculate_input_parameters(self, block_size = 1000):
        '''
        Compute input state from files.
        Parameters
            input_folder: str
                Folder where input state files are located
            block_size: int
                Size for which data is going to be divided for analysis
        Return
            input_alpha: float
                Amplitude of input state
        '''
        print('CALCULATING INPUT STATE PARAMETERS')
        # Import files
        input_data, samplehold, input_meta = self.import_input_files()
        input_phases = ['000', '090']

        #Time information
        input_sequences = input_meta['subarray_count']
        input_total_sequences = len(input_data['000'])
        input_files_number = int(input_total_sequences/input_sequences)

        input_points_per_seq = int(input_meta['wave_array_count']/input_sequences)
        Ts = input_meta['horiz_interval']
        begin = input_meta['horiz_offset']
        total_points = input_meta['wave_array_count']
        x = np.linspace(begin, begin + Ts*total_points, total_points)
        x_sequence = np.linspace(begin, begin + Ts*input_points_per_seq, input_points_per_seq)

        # Separate input state and vacuum data
        reference = samplehold['000'].mean((0,1))
        vac_indices = {phase: [] for phase in input_phases}
        input_indices = {phase: [] for phase in input_phases}
        for phase in input_phases:
            for sequence in range(input_total_sequences):
                if samplehold[phase][sequence].mean() > reference:
                    vac_indices[phase].append(sequence)
                if samplehold[phase][sequence].mean() < reference:
                    input_indices[phase].append(sequence)
            print(len(vac_indices[phase])/input_total_sequences, 'vacuum files and', len(input_indices[phase])/input_total_sequences, 'input files')
            vac_indices[phase] = np.array(vac_indices[phase])
            input_indices[phase] = np.array(input_indices[phase])

        amount_of_blocks = int(input_total_sequences/block_size)
        block_alpha = []
        block_angle = []
        for block in range(amount_of_blocks):
            mf_vac = {phase: [] for phase in input_phases}
            mf_input = {phase: [] for phase in input_phases}
            for phase in input_phases:
                min_point = block*block_size
                max_point = (block+1)*block_size
                min_vac = vac_indices[phase][vac_indices[phase] > min_point].min()
                min_vac_index = np.where(vac_indices[phase] == min_vac)[0][0]
                max_vac = vac_indices[phase][vac_indices[phase] < max_point].max()
                max_vac_index = np.where(vac_indices[phase] == max_vac)[0][0]
                min_input = input_indices[phase][input_indices[phase] > min_point].min()
                min_input_index = np.where(input_indices[phase] == min_input)[0][0]
                max_input = input_indices[phase][input_indices[phase] < max_point].max()
                max_input_index = np.where(input_indices[phase] == max_input)[0][0]

                reduced_vac = []
                reduced_input = []
                for i, ind in enumerate(vac_indices[phase][min_vac_index:max_vac_index]):
                    reduced_vac.append(input_data[phase][ind])
                for i, ind in enumerate(input_indices[phase][min_input_index:max_input_index]):
                    reduced_input.append(input_data[phase][ind])

                _mf_vac = np.zeros(len(reduced_vac))
                _mf_input = np.zeros(len(reduced_input))
                for k in range(len(reduced_vac)):
                    _mf_vac[k] = np.dot(reduced_vac[k], self.mf)
                for k in range(len(reduced_input)):
                    _mf_input[k] = np.dot(reduced_input[k], self.mf)

                mean_vac = np.mean(_mf_vac)
                var_vac = np.var(_mf_vac)

                mf_vac[phase] = (_mf_vac - mean_vac)/np.sqrt(2*var_vac)
                mf_input[phase] = (_mf_input - mean_vac)/np.sqrt(2*var_vac)

            q_mean = np.mean(mf_input['000'])
            p_mean = np.mean(mf_input['090'])

            angle_rad = np.angle(q_mean+1j*p_mean)
            angle_deg = angle_rad*180/np.pi
            ### Take a better look into this
            input_displacement = np.exp(1j*angle_rad)/np.sqrt(2)*(q_mean+1j*p_mean)
            block_alpha.append(np.abs(input_displacement))
            block_angle.append(angle_deg)

            print('Block', block, 'Input |\\alpha| =', block_alpha[-1])
        input_alpha = np.mean(block_alpha)
        input_angle = np.mean(block_angle)
        print('Input |\\alpha| averaged through all blocks: %.2f (%.2f)' %(input_alpha, np.std(block_alpha)))
        print('Input angle averaged through all blocks: %.2f (%.2f)°' %(input_angle, np.std(block_angle)))

        return input_alpha, input_angle

    def reconstruct_input_state(self):
        self.input_alpha, self.relay_theta, self.input_angle = self.read_folder()

        # Construct density matrix of input state
        input_psi = functions.psi_coherent(self.input_alpha, self.relay_theta + self.input_angle, self.dimension)
        input_rho = functions.rho_psi(input_psi)

        # Plots for the input state
        number = np.linspace(0, self.dimension, self.dimension+1)

        print('Input |alpha| = %.2f'%self.input_alpha)
        print('Relay theta = %.2f°'%self.relay_theta)
        print('Input state theta = %.2f°'%self.input_angle)

        print('PLOTS FOR INPUT STATE')
        if self.input_plot:
            # Input density matrix
            plt.figure(figsize=[6.5,6.5])
            plt.imshow(np.abs(input_rho), norm=matplotlib.colors.Normalize(vmin=-np.max(np.abs(input_rho)), vmax=np.max(np.abs(input_rho))))    
            plt.xlabel("n")
            plt.ylabel("m")
            plt.title("Input density matrix")
            #plt.pause(.05)
            # plt.colorbar()
            plt.show()

            # Input photon number distribution
            figure = plt.figure()
            input_photon_number_distribution = [np.abs(input_rho[n,n]) for n in range(self.dimension+1)]
            plt.bar(number, input_photon_number_distribution)
            plt.xlabel("Photon number")
            plt.ylabel("Probability")
            plt.title("Input state photon number distribution")
            #plt.pause(.05)
            plt.show()

            # Wigner function of the input state
            # Calculate Wigner function
            Q, P, W = functions.wigner_function(input_rho, self.dimension)

            # Plot Wigner function
            W_max = np.max(np.abs(W))

            # Countour plot of Wigner function
            fig = plt.figure(figsize=[9,9])
            ax = plt.axes()
            ax.contourf(Q, P, W, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
            ax.set_title('Input state Wigner function')
            ax.set_xlabel('q')
            ax.set_ylabel('p')
            ax.grid()
            fig.add_axes(ax)
            #fig.colorbar(ax)
            plt.show()

        return input_rho
