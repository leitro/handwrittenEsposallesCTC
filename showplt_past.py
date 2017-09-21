import matplotlib.pyplot as plt

train = open('train_cer.log', 'r')
test = open('test_cer.log', 'r')

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
