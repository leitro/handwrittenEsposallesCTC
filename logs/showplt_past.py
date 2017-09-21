import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
    print('Usage: python3 showplt_past.py <train_log> <test_log>')
    exit()
train = open(sys.argv[1], 'r')
test = open(sys.argv[2], 'r')

train_data = train.read().split(' ')[:-1]
test_data = test.read().split(' ')[:-1]

train_ler = [float(i) for i in train_data]
test_ler = [float(i) for i in test_data]

plt.plot(train_ler, 'r-')
train_spot, = plt.plot(train_ler, 'ro')
plt.plot(test_ler, 'b-')
test_spot, = plt.plot(test_ler, 'bo')
plt.legend([train_spot, test_spot], ['train CER', 'test CER'])
plt.xlabel('epoch')
plt.ylim(0, 1.1)
plt.title('training data and test data character error rate')
plt.show()
train.close()
test.close()
