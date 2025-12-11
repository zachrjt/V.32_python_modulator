# ZT 2025
import numpy as np
import scipy as sp
from random import randint
from collections import deque
import matplotlib.pyplot as plt

OVER_SAMPLE_FACTOR = 20                        # Samples per symbol for oversampling
NUM_STATES = 8                                 # Number of possible V.32 encoder states
BETA = 0.1                                     # Rolloff factor of raised cosine filter      
RAISED_COSINE_FILTER_SPAN = 4                  # Number of symbols to consider in truncated impulse response                                
SCRAMBLER_INIT_CODE = [1 for _ in range(24)]   # Initial seed for scrambler/descrambler
TRACEBACK_LENGTH = 16                          # Depth of Viterbi decoder in # of symbols


class trellis_coded_modulator:
    '''
        Performs the scrambling, encoding, pulse shaping, noise addition, sampling, decoding, and descrambling of data.
        Follows the V.32 standard for 9600 bps using 32 QAM and trellis decoding
    '''
    
    # Variables to hold data for I, Q plotting
    impulseDataI = np.array([])
    impulseDataQ = np.array([])

    sqDataI = np.array([])
    sqDataQ = np.array([])

    # Trellis coded modulation (TCM) constellation mapping for symbols
    tcm_constellation_map = {
                    0b00000: (-4,  1),
                    0b00001: ( 0, -3),
                    0b00010: ( 0,  1),
                    0b00011: ( 4,  1),

                    0b00100: ( 4, -1),
                    0b00101: ( 0,  3),
                    0b00110: ( 0, -1),
                    0b00111: (-4, -1),

                    0b01000: (-2,  3),
                    0b01001: (-2, -1),
                    0b01010: ( 2,  3),
                    0b01011: ( 2, -1),

                    0b01100: ( 2, -3),
                    0b01101: ( 2,  1),
                    0b01110: (-2, -3),
                    0b01111: (-2,  1),

                    0b10000: (-3, -2),
                    0b10001: ( 1, -2),
                    0b10010: (-3,  2),
                    0b10011: ( 1,  2),

                    0b10100: ( 3,  2),
                    0b10101: (-1,  2),
                    0b10110: ( 3, -2),
                    0b10111: (-1, -2),

                    0b11000: ( 1,  4),
                    0b11001: (-3,  0),
                    0b11010: ( 1,  0),
                    0b11011: ( 1, -4),

                    0b11100: (-1, -4),
                    0b11101: ( 3,  0),
                    0b11110: (-1,  0),
                    0b11111: (-1,  4),
                    }

    # Shortned, prefix only form of the constellation mapping
    prefix_groups = {0b000: [(-4,  1),( 0, -3),( 0,  1),( 4,  1)],
                    0b001: [(4,  -1),( 0, 3),( 0,  -1),(-4,  -1)],
                    0b010: [(-2,  3), (-2, -1),( 2,  3), ( 2, -1)],
                    0b011: [(2,  -3), (2, 1),(-2, -3), (-2, 1)],
                    0b100: [(-3, -2),( 1, -2),(-3,  2),( 1,  2)],
                    0b101: [(3, 2),( -1, 2),(3,  -2),(-1,  -2)],
                    0b110: [( 1,  4),(-3,  0),( 1,  0),( 1, -4)],
                    0b111: [( -1,  -4),(3,  0),( -1,  0),( -1, 4)]
                }

    # LUT for valid transistions to state variables S0,S1,S2 from (Previous 0bS0S1S2, Y0Y1Y2) values
    valid_transistions = {0b000:[(0b000,0b000), (0b001,0b010), (0b010,0b011), (0b011, 0b001)],
                          
                          0b001:[(0b100,0b100), (0b101,0b111), (0b110,0b110), (0b111, 0b101)],

                          0b010:[(0b000,0b010), (0b001,0b000), (0b010,0b001), (0b011, 0b011)],

                          0b011:[(0b100,0b111), (0b101,0b100), (0b110,0b101), (0b111, 0b110)],

                          0b100:[(0b000,0b011), (0b001,0b001), (0b010,0b000), (0b011, 0b010)],

                          0b101:[(0b100,0b101), (0b101,0b110), (0b110,0b111), (0b111, 0b100)],

                          0b110:[(0b000,0b001), (0b001,0b011), (0b010,0b010), (0b011, 0b000)],

                          0b111:[(0b100,0b110), (0b101,0b101), (0b110,0b100), (0b111, 0b111)]
                          }

    # Empty LUT for fast lookup of nearest 8 unique prefix symbols
    quadrant_region_lookup_table = {1:{},2:{},3:{},4:{}}

    # Encoder delay states
    S = [0b0, 0b0, 0b0]
    Sp = [0b0, 0b0, 0b0]
    Y = [0b0, 0b0, 0b0, 0b0, 0b0]  # current encodedInputput for a nibble
    Yp = [0b0, 0b0, 0b0, 0b0, 0b0] # previous encodedInputput for a nibble

    def __init__(self):
        # Generate the LUT for decoding symbols
        self.initialize_nearest_unique_prefix_symbol_lut()
    

    def initialize_nearest_unique_prefix_symbol_lut(self):
        '''
            Populates a lookup table that allows for fast determination of the 8
            closest unique prefix symbols for a sample based on quadrant (1-4) and region (1-13) 
        '''
        # Test points, ordered 1 for each of the 13 regions
        test_data = [(.5,.5),(1.5,.5),(2.5,.5),
                     (.5,1.5),(1.5,1.5),(.5,2.5),
                     (3.5,1.5),(1.5,3.5),(3,3),
                     (1.5,2.25),(2.5,1.75),
                     (3.5,2.25),(2.5,3.75)]

        for i, quadrant in enumerate([1,2,3,4]):

            for region, point in enumerate(test_data,1):
                # test each point/region

                # calculate sign of the point's x,y using quadrant value
                xsign = -1 if quadrant in (2, 3) else 1
                ysign = -1 if quadrant in (3, 4) else 1
                point_x = xsign*point[0]
                point_y = ysign*point[1]
                
                closest_eight_symbols = np.zeros(NUM_STATES,int)

                # iterate over each of the NUM_STATES prefix groups 4 points, find closest one
                for j, prefix in enumerate(self.prefix_groups.keys()):
                    min_k_value = 0
                    min_distance_value = None
                    for k, symbol in enumerate(self.prefix_groups[prefix]):
                        symbol_distance = np.sqrt(((symbol[0]-point_x)**2)+((symbol[1]-point_y)**2))
                        # print(f"Symbol {((prefix << 2) + k):>05b} - {symbol_distance:.4f}")
                        if k==0:
                            min_distance_value = symbol_distance
                            min_k_value = k
                        elif symbol_distance < min_distance_value:
                            min_k_value = k
                            min_distance_value = symbol_distance

                    # Determined the closest symbol in the prefix group, calculate the symbol value using k (2 LSB) and prefix (3 MSB)
                    closest_eight_symbols[j] = (prefix << 2) + min_k_value
                
                # Populate the closest symbols per prefix group (NUM_STATES) lookup table
                self.quadrant_region_lookup_table[quadrant][region] = closest_eight_symbols.copy()


    def scramble(self, inputData):
        '''
            Scrambles binary data using GPC = 1 + x^18 + x^23 per ITU V.32
        '''

        # Sramble data using "Call mode modem generating polynomial: (GPC)
        scrambledData = []
        shiftRegister = deque(SCRAMBLER_INIT_CODE, maxlen=23)
        
        for byte in inputData:
            scrambledByte = 0x00

            for i in range(8):
                scrambledByte = scrambledByte << 1

                feedback = 0b1 ^ shiftRegister[17] ^ shiftRegister[22]
                scrambled = ((byte & 0x80)>>7) ^ feedback

                scrambledByte += scrambled
                shiftRegister.appendleft(scrambled)
                
                byte = byte << 1

            scrambledData.append(scrambledByte)

        return scrambledData
    

    def descramble(self, inputData):
        '''
            Descrambles decoded data using GPC = 1 + x^18 + x^23 per ITU V.32
        '''
        encodedInputputData = []
        shiftRegister = deque(SCRAMBLER_INIT_CODE, maxlen=23)
        
        for byte in inputData:
            descrambledByte = 0x00

            for i in range(8):
                descrambledByte = descrambledByte << 1

                feedback = 0b1 ^ shiftRegister[17] ^ shiftRegister[22]
                scrambledBit = ((byte & 0x80)>>7) 
                descrambled = scrambledBit ^ feedback

                descrambledByte += descrambled
                shiftRegister.appendleft(scrambledBit)
                
                byte = byte << 1

            encodedInputputData.append(descrambledByte)

        return encodedInputputData
    

    def convolution_encoder(self, inputData):
        '''
            Ingest byte data, and perform encoding as described in V.32 using a differential encoder 
            followed by a 3-state convolutional encoder yield 5 bits for every 4 data bits. 
            Returns the encoded data in reversed format (LSB->MSB) suitable to directly mapping symbols
        '''
        encodedInputputData = []
        for datum in inputData:
            # Extract the upper and lower nibbles
            datum2 = (datum & 0xF0)>>4 # upper/second 4-bits
            datum1 = datum & 0x0F # lower/first 4-bits
            
            for datumTarget in (datum1, datum2):
                # Bit 3,4 of the nibble are passed through
                self.Y[4] = (datumTarget & 0b1000) >> 3
                self.Y[3] = (datumTarget & 0b0100) >> 2

                Q_1 = (datumTarget & 0b0001)
                Q_2 = (datumTarget & 0b0010)>>1

                Y_1 = Q_1 ^ self.Yp[1]
                Y_2 = ((Q_1 & self.Yp[1]) ^ self.Yp[2]) ^ Q_2

                self.Y[2] = Y_2
                self.Y[1] = Y_1
                
                # shift delay states and perform the logic required
                self.S[1] = (self.Sp[2] ^ (Y_1 ^ Y_2)) ^ (self.Sp[0] & (self.Sp[1] ^ Y_2))
                self.S[2] = self.Sp[0]
                self.S[0] = (self.Sp[1] ^ Y_2) ^ (self.Sp[0] & Y_1)
                self.Y[0] = self.Sp[0]
                
                encodedInputputHex = 0x00
                for i in range(5):
                    # bit order is reversed (Y0, Y1, Y2, Q3, Q4) for constellation symbols
                    encodedInputputHex |= (self.Y[i] << (4-i))
                encodedInputputData.append(encodedInputputHex)

                # Update previous delay and data states for next nibble
                self.Sp = self.S[:]
                self.Yp = self.Y[:]

        return encodedInputputData
    

    def constellation_mapper(self, symbolData):
        '''
            Ingests coded symbol values and returns constellation I, Q coordinates
        '''
        encodedInputputPoints = np.empty((len(symbolData)), complex)

        for i in range(len(symbolData)):
            encodedInputputPoints[i] = (self.tcm_constellation_map[symbolData[i]][0] + 1j*self.tcm_constellation_map[symbolData[i]][1])
        return encodedInputputPoints
    

    def sample(self, tx_symbolData, applyNoise=False, SNR=25):
        '''
            Takes in symbol I, Q values, pads them to oversample, pulse shapes them with a raised cosine filter, then performs down sampling and noise addition
        '''
        # Pad the symbol I and Q data, creating a impulse representation of the symbol stream to then filter
        for symbol in tx_symbolData:
            pulseI = np.zeros(OVER_SAMPLE_FACTOR) # to convolute with filter need only impulse response with padding no level-holding square wave
            pulseQ = np.zeros(OVER_SAMPLE_FACTOR)
            
            pulseI[0] = symbol.real
            pulseQ[0] = symbol.imag
            
            self.impulseDataI = np.concatenate((self.impulseDataI, pulseI))
            self.impulseDataQ = np.concatenate((self.impulseDataQ, pulseQ))

            # For the sake of plotting, also creating a level-hold/square wave represenation of the signal
            sqI = np.full(shape=OVER_SAMPLE_FACTOR, fill_value=pulseI[0])
            sqQ = np.full(shape=OVER_SAMPLE_FACTOR, fill_value=pulseQ[0])

            self.sqDataI = np.concatenate((self.sqDataI, sqI))
            self.sqDataQ = np.concatenate((self.sqDataQ, sqQ))

        # Plot the impulse and level-hold versions of the oversampled symbol stream
        # fig, axs = plt.subplots(4)
        # fig.suptitle("IQ data")
        # axs[0].plot(self.impulseDataI, '.-')
        # axs[1].plot(self.sqDataI, '-')

        # axs[2].plot(self.impulseDataQ, '.-')
        # axs[3].plot(self.sqDataQ, '-')

        # axs[0].grid(True)
        # axs[1].grid(True)
        # axs[2].grid(True)
        # axs[3].grid(True)
        # plt.show()

        # Perform pulse shaping using a raised cosine FIR filter
        # If we were dealing with an actual implementation root raised cosine would be used on both the Tx and Rx, 
        # but here we aren't doing any carrier modulation, so a singular raised cosine filtering is done


        number_of_taps = OVER_SAMPLE_FACTOR * RAISED_COSINE_FILTER_SPAN
        t = np.arange(-number_of_taps/2, number_of_taps/2 +1) / OVER_SAMPLE_FACTOR

        h = np.zeros_like(t)

        for i, ti in enumerate(t):
            if np.isclose(ti, 0.0):
                h[i] = 1.0
            elif BETA != 0 and np.isclose(abs(ti), 1/(4*BETA)):
                # limit as t -> ±T/(4β)
                h[i] = (BETA/np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*BETA)) + 
                    (1 - 2/np.pi) * np.cos(np.pi/(4*BETA))
                )
            else:
                numerator = np.sin(np.pi*ti*(1-BETA)) + 4*BETA*ti*np.cos(np.pi*ti*(1+BETA))
                denominator = np.pi*ti*(1 - (4*BETA*ti)**2)
                h[i] = numerator / denominator

        h = h / np.sqrt(np.sum(h**2))   # normalize tap amplitude to have unity power for filter

        i_shaped = np.convolve(self.impulseDataI, h, mode='full')
        q_shaped = np.convolve(self.impulseDataQ, h, mode='full')
        baseband_shaped = i_shaped + 1j*q_shaped

        # Now add noise to the baseband signal

        baseband_shaped_noise = baseband_shaped.copy()

        # apply noise to result
        if applyNoise:
            # calculate signal power
            signalPower = np.mean(np.abs(baseband_shaped)**2)
            snrLinear = 10**(SNR/10)
            # determine awgn noise std
            awgn_power = signalPower / snrLinear
            noise_std = np.sqrt(awgn_power/2)
            noise = noise_std * (np.random.randn(len(baseband_shaped)) + 1j*np.random.randn(len(baseband_shaped)))

            baseband_shaped_noise = baseband_shaped + noise

        # Now performed sampling, here no real "synchronization" is done, we know the exact offset to sample
        # In an actual implementation we would need to implement that which is of interest to attempt as well as channel equalization

        baseband_sampled = np.empty((len(tx_symbolData)), dtype=np.complex128)
        baseband_sampled_noise = np.empty((len(tx_symbolData)), dtype=np.complex128)

        # Sample the data at the known offset times
        delay_offset = (number_of_taps-1)//2
        
        for i in range(len(tx_symbolData)):
            baseband_sampled[i] = baseband_shaped[(i*OVER_SAMPLE_FACTOR)+delay_offset]
            if applyNoise:
                baseband_sampled_noise[i] = baseband_shaped_noise[(i*OVER_SAMPLE_FACTOR)+delay_offset]

        # Plot the shaped data and where optimal sampling would be
        # fig, axs = plt.subplots(2)
        # axs[0].plot(baseband_shaped_noise.real, '-', label="I")
        # axs[1].plot(baseband_shaped_noise.imag, '-', label="Q")
        # axs[0].set_title("Pulse Shaped I Samples with Noise Added")
        # axs[0].set_ylabel("I")
        # axs[1].set_ylabel("Q")
        # axs[0].set_title("Pulse Shaped Q Samples with Noise Added")
        # axs[0].set_xlabel("Sample")
        # axs[1].set_xlabel("Sample")
        # axs[0].grid(True)
        # axs[1].grid(True)

        # for i in range(len(tx_symbolData)):
        #     axs[0].plot([i*OVER_SAMPLE_FACTOR+delay_offset,i*OVER_SAMPLE_FACTOR+delay_offset], [0, i_shaped[i*OVER_SAMPLE_FACTOR+delay_offset]], c="red")
        #     axs[1].plot([i*OVER_SAMPLE_FACTOR+delay_offset,i*OVER_SAMPLE_FACTOR+delay_offset], [0, q_shaped[i*OVER_SAMPLE_FACTOR+delay_offset]], c="red")
        # plt.show()
        # plt.grid(True)
        
        if applyNoise:
            return baseband_sampled_noise
        else:
            return baseband_sampled
        

    def decode(self, sampleData, rescale=False, plotConstellation=False):
        '''
            Performs trellis decoding and subsequent differential decoding of symbol stream, implementing the Viterbi algorithm
        '''

        # First perform a psuedo-AGC function by scaling to have a given RMS power, 
        # This assumes that we have enough samples such that all constellation points are represented
        if rescale:
            sampleRMS = np.sqrt(np.sum((1/len(sampleData)*(np.abs(sampleData)**2))))
            encoderRMS = np.empty(shape=len(self.tcm_constellation_map.keys()),dtype=float)

            for i, value in enumerate(self.tcm_constellation_map.values()):
                encoderRMS[i] = np.sqrt((value[0]**2)+(value[1]**2))
            
            encoderRMS = np.sqrt(np.sum((1/len(encoderRMS))*encoderRMS**2))
            rmsScalingFactor = encoderRMS/sampleRMS

            sampleData = sampleData*rmsScalingFactor
        if plotConstellation:
            # Plot rescaled constellation with actually symbol points overlaid
            plt.scatter(sampleData.real, sampleData.imag, c='blue', marker="." , label="Rx Symbol")
            for k, value in enumerate(self.tcm_constellation_map.values()):
                if k==0:
                    plt.scatter(value[0], value[1], c='red', marker="x", label="Actual Symbol")
                plt.scatter(value[0], value[1], c='red', marker="x")
            
            plt.grid(True)
            plt.title('V.32 32-TCM Constellation')
            plt.ylabel("Q")
            plt.xlabel("I")
            plt.legend()
            plt.show()

        # Initilize tables and variables for decoding
        branch_distances = np.zeros(shape=NUM_STATES,dtype=float)
        path_metric = np.full(NUM_STATES, np.inf)
        path_metric[0] = 0
        survivor = [deque(maxlen=TRACEBACK_LENGTH) for _ in range(NUM_STATES)]
        rx_symbols = []

        ## Start of VITERBI algorithm implementation for V.32
        for i in range(len(sampleData)):
            
            # 1. Determine quadrant and region of the sample
            quad = None
            x = np.real(sampleData[i])
            y = np.imag(sampleData[i])

            if x >=0 and y >=0:
                quad=1
            elif x >=0 and y <=0:
                quad=4
            elif x <=0 and y >=0:
                quad=2
            else:
                quad=3

            xmag =  np.abs(x)
            ymag = np.abs(y)
            region = None

            # region identification code taken from: https://www.ti.com/lit/an/spra099/spra099.pdf
            if xmag <=1:
                if ymag <=1:
                    region=1
                else:
                    if ymag <=2:
                        region=4
                    else:
                        region=6
            else:
                if xmag <= 2:
                    if ymag <=1:
                        region=2
                    else:
                        if ymag <=2:
                            region=5
                        else:
                            if ymag <= xmag+1:
                                region=10
                            else:
                                region=8
                else:
                    if ymag <=1:
                        region=3
                    else:
                        if ymag > xmag+1:
                            region=13
                        else:
                            if ymag <= xmag-1:
                                if ymag <=2:
                                    region=7
                                else:
                                    region=12
                            else:
                                if ymag <=2:
                                    region=11
                                else:
                                    region=9

            # 2. Using the region and quadrant, lookup the 8 closest possible symbols with unique prefixes (Y0,Y1,Y2)
            closest_symbols = self.quadrant_region_lookup_table[quad][region]

            # 2a. Calculate the distance from the sampled symbol to each of the 8 closest unqiue prefix symbols, aka the "branch distances" 
            for j in range(len(closest_symbols)):
                branch_distances[j] = (self.tcm_constellation_map[closest_symbols[j]][0]-x)**2 + (self.tcm_constellation_map[closest_symbols[j]][1]-y)**2

            # 3. Recalculate the path metric for each of the 8 possible states of (S0,S1,S2) using the calculated branch distances and permissible state transisions

            new_path_metric = np.full(NUM_STATES, np.inf)
            new_survivor = [deque(maxlen=TRACEBACK_LENGTH) for _ in range(NUM_STATES)]

            
            for s in [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111]:
                # 4. For each of the given states considered for sample N:
                # 5. Fetch the 4 possible transistions to that state describing what the possible combos of start/previous states and the data/prefix of the transistion/branch
                past_to_current_trans = self.valid_transistions[s][:]
                
                # initialize some temp variables to determine the minimimum distance of the 4 possible transistions
                min_dist = 0
                min_symbol = None
                min_prefix = None
                min_previous_state = 0
                temp_dist = 0

                for k, transistion in enumerate(past_to_current_trans):
                    # 6. For each of the kth possible transistion to state s, calculate the distance using the path metric of the start/previous state 
                    #       and the transistion branch distance found previously

                    # Recall:
                    # - transistion[1] represents the data / symbol prefix: Y0,Y1,Y2
                    # - transistion[0] represents the start / previous state: S0,S1,S2
                    # - closest_symbols[] and branch_distances[] are ordered based on the 2 LSB of the symbol

                    # print(f"previous state: {transistion[0]:03b} metric is {path_metric[transistion[0]]:.3f} calculated distance is {branch_distances[transistion[1]]:.3f}")
                    temp_dist = path_metric[transistion[0]] + branch_distances[transistion[1]]
                    
                    if k == 0:
                        min_dist = temp_dist
                        min_previous_state = transistion[0]
                        min_symbol = closest_symbols[transistion[1]]
                        min_prefix = transistion[1]
                    
                    if temp_dist < min_dist:
                        min_dist = temp_dist
                        min_previous_state = transistion[0]
                        min_symbol = closest_symbols[transistion[1]]
                        min_prefix = transistion[1]

                # print(f"for state {s:03b} - min dist: {min_dist:.3f} - prior state: {min_previous_state} via {min_prefix:03b} -symbol {min_symbol:05b}")

                # 6. Having found the minimum possible distance for the s state of sample N, store the newly calculate distance/path metric
                new_path_metric[s] = min_dist

                # 6a. Copy the survivor path to the determined minimum distance previous state, then update the path with the new branch/prefix symbol to the current state s
                new_survivor[s] = deque(survivor[min_previous_state], maxlen=TRACEBACK_LENGTH)
                new_survivor[s].append(min_symbol)
                

            # 7. After considering all 8 possible states for sample N, update the path metric and survivor tracing variables with the ones recalculated
            path_metric = new_path_metric.copy()
            survivor = [deque(d, maxlen=TRACEBACK_LENGTH) for d in new_survivor]

            # 8. After TRACEBACK_LENGTH number of samples, our the algorithm results in the most probable path as the survivor path that leads to the state with the lowest path metric
            if i >= TRACEBACK_LENGTH -1:
                # 8a. Extract the symbol from the survivor path with the lowest path metric
                optimal_state = np.argmin(path_metric)
                symbolBits = survivor[optimal_state][0]
                rx_symbols.append(symbolBits)
                survivor[optimal_state].popleft()

            # 9. At the end of block of symbols, empty the survivor path contents of the optimal path since the delay is TRACEBACK_LENGTH and otherwise would be left in the sliding window
            if i == len(sampleData)-1:
                # remove remaining data since we have reached the end of the stream
                optimal_state = np.argmin(path_metric)

                for bits in survivor[optimal_state]:
                    rx_symbols.append(bits)
            
        ## End of VITERBI algorithm implementation for V.32

        # Using the determined symbol values, decode the data using the differential decoder
        # Note that there are only 4 data bits in each 5 bit symbol
        # Symbol bit order (LSB->MSB) is swapped with data bit order (MSB->LSB)

        decoded_bytes = []
        # Decoder delay state variables
        Y2_p = 0x00
        Y1_p = 0x00
        working_byte = 0x00


        for k, symbol in enumerate(rx_symbols):
            # for each symbol we have 4 data bits, so every 2 symbols is 1 byte
            # (Y0 Y1 Y2 Q3 Q4)
            # print(f"Input Symbol: {symbol:05b}")

            nibble = 0x00
            # discard MSB - Y0
            # Q3, Q4 can be directly added to the nibble
            Y1 = (symbol & 0b1000)>>3
            Y2 = (symbol & 0b0100)>>2
            Q3 = (symbol & 0b0010)>>1
            Q4 = (symbol & 0b0001)

            # note order of operation here is slightly different than the encoder to reverse the operation
            Q1 = (Y1) ^ (Y1_p)
            Q2 = Y2 ^ Y2_p ^ (Q1 & Y1_p)

            nibble = (Q4 << 3) | (Q3 << 2) | (Q2 << 1) | Q1
            Y1_p = Y1
            Y2_p = Y2

            if (k+1) % 2 == 0:
                # Append the byte to the decoded data array after handling 2 symbols
                working_byte |= (nibble << 4)
                decoded_bytes.append(working_byte)
                # print(f"Overall Byte {working_byte:08b}\n")
                working_byte = 0x00
            else:
                working_byte |= nibble

        return rx_symbols, decoded_bytes


def main():

    SNR = 18
    coder = trellis_coded_modulator()
    
    randData = []
    for i in range(4096):
        randData.append(randint(0,255))

    # Scramble the input data
    scrambledInput = coder.scramble(randData)

    # Encode the input data to get the symbol values
    encodedInput = coder.convolution_encoder(scrambledInput)

    # Lookup the symbol locations to get the I, Q data
    tx_symbolData = coder.constellation_mapper(encodedInput)
    
    # Pad, then oversample the I, Q data performing pulse shaping 
    # Then add noise and downsample the data for the Rx
    tx_sampledData = coder.sample(tx_symbolData, True, SNR)

    # Decode the rx symbol stream using Viterbi algorithm
    rxSymbols, rxData = coder.decode(tx_sampledData, True, True)

    # Descramble the decoded data
    descrambledRxData = coder.descramble(rxData)

    # Calculate symbol and bit error rate statistics
    correctSymbols = 0
    wrongSymbols = 0
    totalSymbols = 0
    correctBits = 0
    wrongBits = 0
    totalBits = 0

    print(f"-----------------------------------------------")
    for k, value in enumerate(encodedInput):
        totalSymbols+=1
        if value == rxSymbols[k]:
            correctSymbols +=1
        else:
            print(f"Input symbol was: {value:05b} rx symbol was {rxSymbols[k]:05b}")
            wrongSymbols +=1
    print()
    for j, value in enumerate(randData):
        totalBits+=8
        if value == descrambledRxData[j]:
            correctBits +=8
        else:
            print(f"Input byte was: {value:08b} rx byte was {descrambledRxData[j]:08b}")
            temp = value ^  descrambledRxData[j]
            for n in range (8):
                if (0x01 & (temp>>n)) == 1:
                    wrongBits +=1
                else:
                    correctBits +=1

    print(f"-----------------------------------------------")
    print(f"Total number of Symbols: {totalSymbols}, Correct: {correctSymbols}, Incorrect: {wrongSymbols}")
    print(f"Symbol Error Rate: {(wrongSymbols/totalSymbols):0.4%}")
    print(f"-----------------------------------------------")
    print(f"Total number of Bits: {totalBits}, Correct: {correctBits}, Incorrect: {wrongBits}")
    print(f"Bit Error Rate: {(wrongBits/totalBits):0.4%}")
    print(f"-----------------------------------------------")
    
    del coder
    
    
    quit()

    # Code below is just generate SNR-SER/BER plots
    SNR_to_test =  np.arange(20, 12, -0.2)
    resulting_SER = []
    resulting_BER = []
    N=10

    for snr in SNR_to_test:
        BER_temp = 0
        SER_temp = 0
        for i in range (N):
            coder = trellis_coded_modulator()
            randData = []
            for i in range(4096):
                randData.append(randint(0,255))
            scrambledInput = coder.scramble(randData)
            encodedInput = coder.convolution_encoder(scrambledInput)
            tx_symbolData = coder.constellation_mapper(encodedInput)
            tx_sampledData = coder.sample(tx_symbolData, True, snr)
            rxSymbols, rxData = coder.decode(tx_sampledData, True, False)
            descrambledRxData = coder.descramble(rxData)
            correctSymbols = 0
            wrongSymbols = 0
            totalSymbols = 0
            correctBits = 0
            wrongBits = 0
            totalBits = 0
            for k, value in enumerate(encodedInput):
                totalSymbols+=1
                if value == rxSymbols[k]:
                    correctSymbols +=1
                else:
                    wrongSymbols +=1
            for j, value in enumerate(randData):
                totalBits+=8
                if value == descrambledRxData[j]:
                    correctBits +=8
                else:
                    temp = value ^  descrambledRxData[j]
                    for n in range (8):
                        if (0x01 & (temp>>n)) == 1:
                            wrongBits +=1
                        else:
                            correctBits +=1
            SER_temp += (wrongSymbols/totalSymbols)
            BER_temp += (wrongBits/totalBits)
            del coder

        print(f"Complete SNR: {snr}dB")
        resulting_BER.append(BER_temp/N)
        resulting_SER.append(SER_temp/N)

    plt.figure()
    plt.semilogy(SNR_to_test, resulting_BER, marker='o')
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both")
    plt.title("BER vs SNR")
    plt.show()

if __name__ == '__main__':
    main()
