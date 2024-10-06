import re

def convert_to_dict(text):
    # Initialize dictionary to store the result
    result = {}

    # Split the text into blocks
    blocks = text.strip().split("\n\n")

    # Process each block
    for block in blocks:
        lines = block.strip().split("\n")
        key = lines[0].strip()  # First line is the key
        epochs = [int(re.search(r"epoch (\d+)", line).group(1)) for line in lines[1:]]
        result[key] = epochs

    return result


input = """
211bfe42
epoch 47500     fits bpd 4.033  val bpd 3.458
epoch 30000     fits bpd 3.938  val bpd 3.769


434423d5
epoch 6920      fits bpd 2.599  val bpd 2.562
epoch 9515      fits bpd 2.598  val bpd 2.569


5a2b4cab
epoch 6920      fits bpd 2.697  val bpd 2.647
epoch 5190      fits bpd 2.736  val bpd 2.660


7230cde8
epoch 17760     fits bpd 3.035  val bpd 2.963
epoch 7770      fits bpd 3.087  val bpd 2.997


839dd6d8
epoch 6055      fits bpd 2.774  val bpd 2.741
epoch 2595      fits bpd 2.807  val bpd 2.766


995c05dd
epoch 15000     fits bpd 8.078  val bpd 8.056
epoch 17500     fits bpd 8.124  val bpd 8.032


af70ee88
epoch 47500     fits bpd 4.599  val bpd 4.297
epoch 43750     fits bpd 4.762  val bpd 4.336


c2ac18ea
epoch 7785      fits bpd 2.566  val bpd 2.506
epoch 17300     fits bpd 2.542  val bpd 2.508


d75d8115
epoch 8650      fits bpd 2.560  val bpd 2.522
epoch 12975     fits bpd 2.538  val bpd 2.491


f41aac70
epoch 27500     fits bpd 2.984  val bpd 4.363
epoch 48750     fits bpd 2.932  val bpd 4.349


06adcf74
epoch 12500     fits bpd 5.590  val bpd 5.587
epoch 5625      fits bpd 5.595  val bpd 5.602


2275c49f
epoch 28750     fits bpd 8.358  val bpd 7.776
epoch 37500     fits bpd 8.168  val bpd 7.690


44a53620
epoch 42500     fits bpd 7.323  val bpd 7.076
epoch 47500     fits bpd 7.594  val bpd 7.031


5af0af3b
epoch 4440      fits bpd 2.643  val bpd 2.592
epoch 3885      fits bpd 2.654  val bpd 2.625


73eae6b8
epoch 28750     fits bpd 4.252  val bpd 3.963
epoch 31250     fits bpd 4.074  val bpd 3.944


850e9da3
epoch 18750     fits bpd 11.104 val bpd 11.107
epoch 25000     fits bpd 11.098 val bpd 11.104


9cf735e9
epoch 35000     fits bpd 4.066  val bpd 3.628
epoch 18750     fits bpd 4.045  val bpd 3.670


b07ac0bf
epoch 4995      fits bpd 2.743  val bpd 2.680
epoch 5550      fits bpd 2.723  val bpd 2.698


c3c7899d
epoch 2220      fits bpd 6.297  val bpd 6.235
epoch 2775      fits bpd 5.823  val bpd 5.780


d7d3a8e5
epoch 45000     fits bpd 8.735  val bpd 8.258
epoch 47500     fits bpd 8.574  val bpd 8.257


f54e2ca3
epoch 6920      fits bpd 2.508  val bpd 2.456
epoch 11245     fits bpd 2.496  val bpd 2.455


0fcfa722
epoch 17205     fits bpd 5.094  val bpd 4.999
epoch 18870     fits bpd 5.107  val bpd 5.019


22fd3cfe  
epoch 18750     fits bpd 5.579  val bpd 5.582
epoch 25000     fits bpd 5.574  val bpd 5.577


48c823ec  
epoch 12210     fits bpd 5.216  val bpd 5.145
epoch 13320     fits bpd 5.219  val bpd 5.125


5c6d9414  
epoch 1665      fits bpd 2.899  val bpd 2.807
epoch 2775      fits bpd 2.798  val bpd 2.713


76017d45  
epoch 68750     fits bpd 11.335 val bpd 11.322
epoch 93750     fits bpd 11.332 val bpd 11.306


8598394e  
epoch 2220      fits bpd 2.738  val bpd 2.669
epoch 2775      fits bpd 2.723  val bpd 2.659


9eac2b31  
epoch 3885      fits bpd 5.732  val bpd 5.683
epoch 4995      fits bpd 5.440  val bpd 5.390


b6458acd  skip


c5874112  
epoch 14705     fits bpd 2.610  val bpd 2.584
epoch 15570     fits bpd 2.601  val bpd 2.584


d91e8c6a  
epoch 3125      fits bpd 5.581  val bpd 5.582
epoch 3750      fits bpd 5.580  val bpd 5.581


f7c12534
epoch 62500     fits bpd 11.130 val bpd 11.128
epoch 112500    fits bpd 11.123 val bpd 11.121


100478ac  
epoch 28545     fits bpd 5.025  val bpd 4.916
epoch 31140     fits bpd 5.047  val bpd 4.906


25d4e024  
epoch 555       fits bpd 2.849  val bpd 2.771
epoch 1110      fits bpd 2.819  val bpd 2.741


4913ab8f  
epoch 22490     fits bpd 4.869  val bpd 4.813
epoch 26815     fits bpd 4.890  val bpd 4.802


5ec01f1d  
epoch 20760     fits bpd 4.921  val bpd 4.825
epoch 34600     fits bpd 4.874  val bpd 4.794


76d59958  
epoch 93750     fits bpd 11.089 val bpd 11.088
epoch 100000    fits bpd 11.088 val bpd 11.089


881eeb38  
epoch 18750     fits bpd 5.580  val bpd 5.574
epoch 50000     fits bpd 5.566  val bpd 5.558


9f516c64  
epoch 19895     fits bpd 2.603  val bpd 2.558
epoch 20760     fits bpd 2.618  val bpd 2.556


b667b5a1  
epoch 48750     fits bpd 7.730  val bpd 6.678
epoch 41250     fits bpd 7.652  val bpd 7.021


cd0b81b1  
epoch 19030     fits bpd 2.595  val bpd 2.564
epoch 19895     fits bpd 2.594  val bpd 2.559


e3ec4898  
epoch 12765     fits bpd 5.247  val bpd 5.143
epoch 13320     fits bpd 5.235  val bpd 5.126


fac7d04c
epoch 16650     fits bpd 5.014  val bpd 4.933
epoch 17760     fits bpd 5.005  val bpd 4.939


12275314  
epoch 3330      fits bpd 6.840  val bpd 6.788
epoch 3885      fits bpd 6.520  val bpd 6.453


2629e0f7  
epoch 28545     fits bpd 4.740  val bpd 4.702
epoch 29410     fits bpd 4.761  val bpd 4.708


4a0d15b8  
epoch 27500     fits bpd 4.161  val bpd 3.764
epoch 17500     fits bpd 4.294  val bpd 3.866


5ee868f8
epoch 4440      fits bpd 2.801  val bpd 2.754
epoch 6660      fits bpd 2.784  val bpd 2.746


76da254d  
epoch 1875      fits bpd 5.622  val bpd 5.611


89935804  
epoch 5550      fits bpd 5.188  val bpd 5.122
epoch 6105      fits bpd 5.120  val bpd 5.036


a0d21a33  
epoch 42500     fits bpd 5.984  val bpd 5.960
epoch 46250     fits bpd 5.995  val bpd 5.963


b89bd3fc  
epoch 3125      fits bpd 5.605  val bpd 5.601


cdc8408d  
epoch 46250     fits bpd 4.381  val bpd 3.982
epoch 26250     fits bpd 4.428  val bpd 4.111


e45f63ab
epoch 42500     fits bpd 2.894  val bpd 3.948
epoch 46250     fits bpd 2.920  val bpd 4.406


14d305be  
epoch 30000     fits bpd 8.313  val bpd 7.653
epoch 31250     fits bpd 8.368  val bpd 7.656


2cb2d77e  
epoch 28750     fits bpd 4.430  val bpd 3.989
epoch 15000     fits bpd 4.312  val bpd 4.018


4e294615  
epoch 36250     fits bpd 7.621  val bpd 7.273
epoch 50000     fits bpd 7.626  val bpd 7.225


60064d3d  
epoch 41250     fits bpd 5.857  val bpd 5.897
epoch 42500     fits bpd 5.859  val bpd 5.884


78a6ee69  
epoch 7785      fits bpd 2.589  val bpd 2.559
epoch 9515      fits bpd 2.583  val bpd 2.555


8e46eeac  
epoch 17205     fits bpd 5.148  val bpd 5.079
epoch 18315     fits bpd 5.144  val bpd 5.065


a19a76fa  
epoch 33750     fits bpd 3.938  val bpd 3.570
epoch 41250     fits bpd 4.029  val bpd 3.413


bbf8a50a  
epoch 25000     fits bpd 3.134  val bpd 3.756
epoch 40000     fits bpd 3.106  val bpd 4.186


ced0293f  
epoch 100000    fits bpd 5.600  val bpd 5.588
epoch 106250    fits bpd 5.592  val bpd 5.585


e6a11b09
epoch 11250     fits bpd 3.103  val bpd 3.739
epoch 42500     fits bpd 2.897  val bpd 3.554


14ed71e9  
epoch 25085     fits bpd 4.819  val bpd 4.743
epoch 28545     fits bpd 4.799  val bpd 4.739


2f8c2bfb  
epoch 20760     fits bpd 4.890  val bpd 4.805
epoch 21625     fits bpd 4.853  val bpd 4.782


5707f69e  
epoch 47500     fits bpd 7.694  val bpd 6.982
epoch 46250     fits bpd 7.817  val bpd 7.086


60ce9e74  
epoch 37500     fits bpd 11.096 val bpd 11.093
epoch 43750     fits bpd 11.097 val bpd 11.095


79f388e7  
epoch 36250     fits bpd 5.918  val bpd 5.964
epoch 48750     fits bpd 5.856  val bpd 5.889


8ed28dc4  
epoch 17760     fits bpd 5.386  val bpd 5.267
epoch 19980     fits bpd 5.394  val bpd 5.305


a215d9dc  
epoch 43750     fits bpd 11.288 val bpd 11.269
epoch 50000     fits bpd 11.274 val bpd 11.265


bce39e80  
epoch 27500     fits bpd 4.018  val bpd 3.468
epoch 37500     fits bpd 4.048  val bpd 3.818


d1cf3a91  
epoch 860       fits bpd 8.261  val bpd 8.216
epoch 865       fits bpd 8.418  val bpd 8.310


e761fe43
epoch 25950     fits bpd 4.923  val bpd 4.823
epoch 27680     fits bpd 4.938  val bpd 4.819


17b54f1c  
epoch 28545     fits bpd 4.801  val bpd 4.737
epoch 31140     fits bpd 4.776  val bpd 4.723


3087adfc  
epoch 12500     fits bpd 11.093 val bpd 11.089
epoch 18750     fits bpd 11.086 val bpd 11.080


5805567e  
epoch 3330      fits bpd 2.636  val bpd 2.597
epoch 3885      fits bpd 2.628  val bpd 2.591


62506018  
epoch 46250     fits bpd 8.500  val bpd 7.796
epoch 38750     fits bpd 8.609  val bpd 8.010


7e323424  
epoch 14430     fits bpd 2.842  val bpd 2.776
epoch 16095     fits bpd 2.832  val bpd 2.758


8f2d74cb  
epoch 18750     fits bpd 4.200  val bpd 4.145
epoch 22500     fits bpd 4.192  val bpd 4.068


a2b9c739  
epoch 43750     fits bpd 8.116  val bpd 7.823
epoch 45000     fits bpd 8.091  val bpd 7.889


bd40e4fb  
epoch 37500     fits bpd 8.053  val bpd 7.729
epoch 43750     fits bpd 7.992  val bpd 7.685


d4939562  
epoch 5550      fits bpd 5.237  val bpd 5.178
epoch 6105      fits bpd 5.178  val bpd 5.112


ef5aec20
epoch 46250     fits bpd 6.123  val bpd 6.228
epoch 48750     fits bpd 6.085  val bpd 6.204


1914f98f  
epoch 8880      fits bpd 3.118  val bpd 3.071
epoch 12765     fits bpd 3.102  val bpd 3.052


34d211da  
epoch 5000      fits bpd 5.714  val bpd 5.710
epoch 5625      fits bpd 5.716  val bpd 5.713


593e3318  
epoch 62500     fits bpd 5.580  val bpd 5.580
epoch 81250     fits bpd 5.575  val bpd 5.572


691fcbf1  
epoch 24220     fits bpd 4.861  val bpd 4.763
epoch 25085     fits bpd 4.840  val bpd 4.757


81c8f988  
epoch 37500     fits bpd 7.618  val bpd 6.903
epoch 35000     fits bpd 7.711  val bpd 7.420


93a713a2
epoch 2500      fits bpd 11.118 val bpd 11.110


a6247be3
epoch 3125      fits bpd 5.592  val bpd 5.587
epoch 3750      fits bpd 5.585  val bpd 5.583


c1db677d
epoch 16435     fits bpd 2.563  val bpd 2.522
epoch 19030     fits bpd 2.553  val bpd 2.513


d4a01100 
epoch 43750     fits bpd 11.091 val bpd 11.085
epoch 56250     fits bpd 11.096 val bpd 11.096


efd9ed19
epoch 5625      fits bpd 11.083 val bpd 11.086
epoch 6250      fits bpd 11.085 val bpd 11.082


1d9a8f8f  
epoch 7770      fits bpd 2.771  val bpd 2.719
epoch 8325      fits bpd 2.774  val bpd 2.727


4296ee8d  
epoch 17205     fits bpd 4.964  val bpd 4.894
epoch 20535     fits bpd 4.964  val bpd 4.858


5a2751e4 
epoch 19895     fits bpd 2.730  val bpd 2.697
epoch 20760     fits bpd 2.717  val bpd 2.694


712e10ac
epoch 12210     fits bpd 2.644  val bpd 2.593
epoch 8880      fits bpd 2.687  val bpd 2.621


8257729c 
epoch 26815     fits bpd 4.807  val bpd 4.738
epoch 31140     fits bpd 4.765  val bpd 4.727


95917c50 
epoch 36250     fits bpd 4.606  val bpd 4.146
epoch 50000     fits bpd 4.555  val bpd 4.103


a9c8d8bb
epoch 21250     fits bpd 4.108  val bpd 3.887
epoch 45000     fits bpd 4.274  val bpd 3.828


c263fad4  
epoch 5000      fits bpd 5.743  val bpd 5.728
epoch 50000     fits bpd 5.725  val bpd 5.718


d5f41138 
epoch 5000      fits bpd 11.101 val bpd 11.099
epoch 6250      fits bpd 11.093 val bpd 11.085


f033cf23
epoch 24220     fits bpd 5.052  val bpd 4.945
epoch 25085     fits bpd 5.073  val bpd 4.947
"""

result = convert_to_dict(input)

cmd = ""
node = 4
for id, epochs in result.items():
    for epoch in epochs:
        cmd += f"python3 idf_2d.py 09_02_23_41 {id} {epoch} {node}\n"
    node = 5 if node == 4 else 4

print(cmd)
print(len(result))

