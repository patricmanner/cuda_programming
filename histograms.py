import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('output.txt',header=None)

data.plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Bins')
plt.title('DD plot')
#plt.savefig('dd_plot')

data2 = pd.read_csv('output2.txt',header=None)

data2.plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Bins')
plt.title('DR plot')
#plt.savefig('dr_plot')

data3 = pd.read_csv('output3.txt',header=None)

data3.plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Bins')
plt.title('RR plot')
#plt.savefig('rr_plot')

data4 = pd.read_csv('output4.txt',header=None)

data4.plot(kind='bar')
plt.ylabel('Omega')
plt.xlabel('Bins')
plt.title('Omega values')
plt.ylim([-1, 2])
#plt.savefig('o_plot')

plt.show()
