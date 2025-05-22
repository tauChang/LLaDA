import matplotlib.pyplot as plt
import collections

x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, \
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
         2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, \
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, \
         3, 3, 3, 3, 3, 3, 3, 4]
# x = [2] * len(x)


plt.plot(x)
plt.xlabel("Step")
plt.ylabel("Num of tokens unmasked per step")
# y lim 0, 1, 2, 3, 4
plt.ylim(0, 4.5)
# ytick integers
plt.yticks(range(0, 5))


# save
plt.savefig("denoising_cosine.png", dpi=300, bbox_inches='tight')
