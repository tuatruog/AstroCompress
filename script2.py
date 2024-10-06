
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def parse_text_to_dict(text):
    lines = text.strip().splitlines()
    result = {}
    # Use a loop to extract pairs of lines and fill the dictionary
    i = 0
    while i < len(lines):
        key = lines[i].strip()
        if key:  # If key is not empty
            i += 1
            value = lines[i].strip() if i < len(lines) else None
            result[key] = value
        i += 1

    final = {}
    for k, v in result.items():
        key = k
        if '_' in k:
            key = k[:k.find('_')]
        if is_float(v):
            final[key] = float(v)

    return final


text = """
12275314
error


62506018_46250
8.710


89935804
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.17 GiB. GPU


79f388e7
n


60064d3d_42500
5.941


bbf8a50a_25000
5.088


f41aac70_27500
4.292


bd40e4fb_43750
8.393


a2b9c739_45000
8.421


b667b5a1
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.44 GiB. GPU


995c05dd
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


14d305be_31250
8.631


d7d3a8e5_47500
8.957


44a53620
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.48 GiB. GPU


81c8f988_37500
8.067


cdc8408d_46250
4.521


73eae6b8_31250
4.389


a19a76fa_41250
4.076


211bfe42_47500
4.159


2cb2d77e_28750
4.521


4a0d15b8_27500
4.443


bce39e80
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.66 GiB. GPU


9cf735e9
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.09 GiB. GPU


c3c7899d_2775
5.864


e3ec4898_13320
5.305


d4939562_5550
117799605.850


d4939562_6105
nan


4296ee8d
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.12 GiB. GPU


48c823ec_13320
5.323


0fcfa722_17205
5.188


fac7d04c_16650
5.090


b07ac0bf_5550
2.738


1914f98f_12765
3.129


712e10ac
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.18 GiB. GPU


5ee868f8
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.14 GiB. GPU


8598394e_2775
2.774


1d9a8f8f_7770
2.827


5805567e_3885
2.677


5af0af3b_4440
2.694


76d59958_100000
11.086


850e9da3
n


93a713a2
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


3087adfc_18750
11.088


a215d9dc_50000
11.270


60ce9e74_37500
11.091


b6458acd
n


efd9ed19_6250
11.079


c263fad4_50000
5.724


0067479a
n


76da254d
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


881eeb38_50000
5.565


06adcf74_12500
5.590


b89bd3fc
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


d91e8c6a_3125
5.580


a6247be3_3750
5.582


5ec01f1d_34600
5.490


e761fe43_27680
5.788


14ed71e9_28545
5.663


8257729c_31140
6.712


691fcbf1_24220
7.031


4913ab8f_22490
6.207


2629e0f7_29410
6.360


d1cf3a91
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


5a2b4cab
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


d75d8115_12975
3.001


5a2751e4_19895
2.874


cd0b81b1_19030
2.738


c1db677d_19030
3.325


434423d5_6920
3.018


839dd6d8_6055
3.090


c5874112_14705
2.921


ef5aec20_48750
6.360


a0d21a33_46250
6.020


e6a11b09_11250
3.734


e45f63ab_46250
4.620


5707f69e_47500
7.970


2275c49f_37500
8.550


4e294615_50000
8.065


95917c50_50000
4.722


a9c8d8bb_45000
4.308


af70ee88_43750
4.851


8f2d74cb
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.61 GiB. GPU


8ed28dc4_19980
5.403


8e46eeac
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.13 GiB. GPU


9eac2b31_4995
5.555


7230cde8_17760
3.051


7e323424
nan


5c6d9414
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


25d4e024_1110
2.879


f7c12534_112500
11.124


d4a01100_56250
11.088


76017d45_93750
11.331


d5f41138_6250
11.092


ced0293f_106250
5.590


593e3318_81250
5.575


34d211da
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


22fd3cfe_25000
5.574


100478ac_31140
5.461


2f8c2bfb
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


f033cf23_24220
6.164


17b54f1c_31140
7.345


f54e2ca3
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


c2ac18ea_17300
2.882


9f516c64_20760
3.163


78a6ee69_9515
3.089
"""


import shlex
def parse_cmd(cmd):
    args = shlex.split(cmd)  # Splits the command string like a shell would
    arg_dict = {}

    key = None
    for arg in args:
        if arg.startswith("--"):
            # If a new flag starts, store the previous key-value pair if exists
            key = arg.lstrip("--")
            # Initialize with True if it's a flag with no value
            arg_dict[key] = True
        elif key:
            # If the flag has a value, update the value in the dictionary
            arg_dict[key] = arg
            key = None

    return arg_dict


import json
from collections import defaultdict
job_mapping_path = f"D:\Documents Tuan/job_mapping.json"
with open(job_mapping_path, 'r') as f:
    job_mapping = json.load(f)

result = parse_text_to_dict(text)
final = defaultdict(list)
for k, v in result.items():
    ds = parse_cmd(job_mapping[k])['dataset']
    final[ds].append(v)


for k, v in final.items():
    high = max(v)
    print(f'Highest for dataset {k} is {high}')

with open("D:\Documents Tuan/job_result.json", 'w') as f:
    json.dump(result, f)

