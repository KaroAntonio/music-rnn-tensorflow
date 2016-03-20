from data_utils.parse_files import *
import numpy as np
'''
def time_blocks_to_fft_blocks(blocks_time_domain):
    fft_blocks = []
    for block in blocks_time_domain:
        fft_block = np.fft.fft(block)
        new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
        fft_blocks.append(new_block)
    return fft_blocks

def fft_blocks_to_time_blocks(blocks_ft_domain):
    time_blocks = []
    for block in blocks_ft_domain:
        num_elems = block.shape[0] / 2
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        new_block = real_chunk + 1.0j * imag_chunk
        time_block = np.fft.ifft(new_block)
        time_blocks.append(time_block)
    return time_blocks
'''
def audio_unit_test(filename, filename2):
    data, bitrate = read_wav_as_np(filename)
    #must be in mono
    if len(data.shape) == 2:
        data = np.squeeze(np.delete(data,-1,1), axis=(1,))
    time_blocks = convert_np_audio_to_sample_blocks(data, 1024)
    ft_blocks = time_blocks_to_fft_blocks(time_blocks)
    ft_blocks = np.array(ft_blocks)
    
    print(ft_blocks.shape)
    time_blocks = fft_blocks_to_time_blocks(ft_blocks)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    write_np_as_wav(song, bitrate, filename2)
    return ft_blocks

ft_blocks = audio_unit_test('data/test/src/Test.wav', 'data/clipped.wav')
