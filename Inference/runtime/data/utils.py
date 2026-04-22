import bchlib
import numpy
import random
import re


str2bytearray = lambda x: bytearray(int(x, 2).to_bytes(length=2, byteorder='little'))
list_int2str = lambda x: [str(bit) for bit in x]
list_str2int = lambda x: [int(bit) for bit in x]


def get_bytes(message):
    
    
    
    
    
    
    
    
    
    
    
    
    data_len = len(message)
    byte_len = int((data_len + 8 - 1) // 8)  
    data_bytes = bytearray(int.to_bytes(int(''.join(message), 2), length=byte_len, byteorder='big'))
    return data_bytes



def unit_encode(message, info_bits, total_bits):
    """
    :param message:输入单个组，如['0', '0', '1', '1', '0', '0', '1', '1', '0', '0', '1', '1', '0', '0', '1', '1']
    :param info_bits:信息位的长度，如16
    :param total_bits:单个组的总长度，如31
    :return信息位，纠错位
    """
    assert total_bits >= 31  
    ecc_bits = total_bits - info_bits
    info_byte_num = int((info_bits + 8 - 1) // 8)  
    ecc_byte_num = int((ecc_bits + 8 - 1) // 8)  
    ecc_pad_bit_num = ecc_byte_num * 8 - ecc_bits

    message = ['0'] * (info_bits - len(message)) + message  
    data_bytes = bytearray(
        int.to_bytes(int(''.join(message), 2), length=info_byte_num, byteorder='big'))  
    ecc = bch.encode(data_bytes)  

    ecc_message = bin(int.from_bytes(ecc, byteorder='big'))[2:ecc_bits+2-ecc_pad_bit_num]  
    ecc_data = ecc_message.zfill(ecc_bits)  
    return list_str2int(message), list_str2int(ecc_data)



def unit_decode(message, info_bits, total_bits):
    """
    :param message:输入单个组
    :param info_bits:信息位的长度，如16
    :param total_bits:单个组的总长度，如31
    :return解码后的信息位
    """
    ecc_bits = total_bits - info_bits
    info_byte_num = int((info_bits + 8 - 1) // 8)  
    ecc_byte_num = int((ecc_bits + 8 - 1) // 8)  
    ecc_pad_bit_num = ecc_byte_num * 8 - ecc_bits

    info_data = message[:info_bits]  
    info_data = list_int2str(info_data)
    ecc_data = message[info_bits:]  
    ecc_data = ecc_data + [0] * ecc_pad_bit_num  
    ecc_data = list_int2str(ecc_data)

    info_data_bytes = bytearray(
        int.to_bytes(int(''.join(info_data), 2), length=info_byte_num, byteorder='big'))  
    ecc_data_bytes = bytearray(
        int.to_bytes(int(''.join(ecc_data), 2), length=ecc_byte_num, byteorder='big'))  
    bitflips, c_data, c_ecc = bch.decode(info_data_bytes, ecc_data_bytes)  

    decoded_message = bin(int.from_bytes(c_data, byteorder='big'))[2:]  
    decoded_message = decoded_message.zfill(info_bits)  
    return list_str2int(decoded_message)


def encode(total_message, unit_info_bits, unit_total_bits):
    """
    :param total_message:需要生成纠错的全部数据
    :param unit_info_bits:单元内信息位的长度，如16
    :param unit_total_bits:单元总长度，如31
    :return分组加上纠错位后的数据
    """
    message_len = len(total_message)
    out_data = []
    for split_index in range(0, message_len, unit_info_bits):
        if split_index + unit_info_bits <= message_len:
            split = total_message[split_index: split_index + unit_info_bits]
            split, ecc_data = unit_encode(list_int2str(split), info_bits=unit_info_bits, total_bits=unit_total_bits)
            out_data = out_data + split + ecc_data
        else:
            split = total_message[split_index:]
            out_data = out_data + split  
    return out_data


def decode(extracted_data, unit_info_bits, unit_total_bits):
    
    message_len = len(extracted_data)
    extracted_data = list_int2str(extracted_data)
    out_data = []
    
    for split_index in range(0, message_len, unit_total_bits):
        if split_index + unit_total_bits <= message_len:
            split = extracted_data[split_index: split_index + unit_total_bits]
            decoded_data = unit_decode(split, info_bits=unit_info_bits, total_bits=unit_total_bits)
            out_data = out_data + decoded_data
        else:
            split = extracted_data[split_index:]
            out_data = out_data + split
    return out_data



BCH_POLYNOMIAL = 299
BCH_BITS = 22  
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)  
if __name__ == "__main__":
    count = 0
    for i in range(100):
        data = numpy.random.randint(low=0, high=2, size=64).tolist()
        print("Original data:", data)
        print("Data Length:", len(data))
        encoded = encode(data, unit_info_bits=64, unit_total_bits=240)
        print("Original encoded data:", encoded)
        print("Original encoded data Length:", len(encoded))

        
        for _ in range(30):
            flip_index = random.randint(0, len(encoded) - 1)
            encoded[flip_index] = 1 - encoded[flip_index]

        
        decoded_flipped = decode(encoded, unit_info_bits=64, unit_total_bits=240)
        print("Decoded data Length:", len(decoded_flipped))
        print("Decoded data after flipping 3 bits:", decoded_flipped)

        flag = True
        for i in range(64):
            if decoded_flipped[i] != data[i]:
                flag = False
        count += 1 if flag else 0
    print(count)

    
    
    
    
    
    
    

    
    
    

    
    
    
    
    
    
    
