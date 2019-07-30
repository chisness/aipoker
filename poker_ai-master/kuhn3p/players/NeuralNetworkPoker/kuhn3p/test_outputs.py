import betting

states = {
    'c':       1, # k
    'cc':      1,  # kk
    'ccr':     2, # kkb
    'r':       2, # b    
    'cr':      2, # kb  
    'crf':     3, # kbf      
    'ccrf':    3, # kkbf
    'rf':      3, # bf  
    'crc':     4, # kbc      
    'ccrc':    4, # kkbc
    'rc':      4, #bc
    'ccc':     'ikkk',    
    'crcc':    'ikbcc',
    'ccrcf':   'ikkbcf',
    'ccrcc':   'ikkbcc',
    'ccrfc':   'ikkbfc',
    'ccrff':   'ikkbff',
    'crfc':    'ikbfc',
    'crff':    'ikbff',
    'crcf':    'ikbcf',
    'rfc':     'ibfc',
    'rcc':     'ibcc',
    'rcf':     'ibcf',
    'rff':     'ibff',
}


for k in states:
    print(k, betting.string_to_state(k))