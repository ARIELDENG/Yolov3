# import matplotlib.pyplot as plt
#
# f = open('./log.txt','r')
# loss = []
#
# for line in f.readlines():
#     line1 = line.split('||')[0]
#     loss.append(line1[-9:-1])
#
# plt.plot(range(len(loss)), loss)
# plt.show()