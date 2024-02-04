

file = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/train_data.txt"
new_file = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/BERt/train_data.txt"
lines = []
with open(file, 'r') as f:
    lines= f.readlines()
f.close()
with open(new_file, 'w') as nf:
    for line in lines:
        
        line = line.replace(' ::: ', ',')
         
        nf.writelines(line)
nf.close()
