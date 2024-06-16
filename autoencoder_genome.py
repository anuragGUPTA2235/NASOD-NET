import random as rd

file_path = "bioex/autoencoder_genome.txt"
with open(file_path,'a') as file:
 for items in range(0,1000):
  string = ""
  p1 = rd.randint(0,6)
  p2 = rd.randint(0,5)
  p3 = rd.randint(0,4)
  p4 = rd.randint(0,3)
  p5 = rd.randint(0,2)
  p6 = rd.randint(0,1)
  q1 = rd.randint(0,6)
  q2 = rd.randint(0,5)
  q3 = rd.randint(0,4)
  q4 = rd.randint(0,3)
  q5 = rd.randint(0,2)
  q6 = rd.randint(0,1)
  r1 = rd.randint(0,6)
  r2 = rd.randint(0,5)
  r3 = rd.randint(0,4)
  r4 = rd.randint(0,3)
  r5 = rd.randint(0,2)
  r6 = rd.randint(0,1)  
  string = string + str(p1) + str(p2) + str(p3) + str(p4) + str(p5) + str(p6) + str(q1) + str(q2) + str(q3) + str(q4) +    str(q5) + str(q6) + str(r1) + str(r2) + str(r3) + str(r4) + str(r5) + str(r6)
  file.write(string)
  file.write("\n")

print("genome dataset for autoencoder created successfully")  