import numpy as np



def cluster(kjz1):
    """
    perform a means 2 cluster. e.g 2-cluster approach,
    Compute a mean of whole data.assign a class smaller than mean,while other class greater.
    Recompute means of each classes,Taken the mean of both class as new mean,and redo it
    until it not changed.
    :param kjz1:
    :return:
    """

    bj = 1
    kjz1 = np.sort(kjz1)
    while (True):
        if bj == 1:
            kj = np.mean([kjz1[0], kjz1[len(kjz1) - 1]])  # 初始分组均值使用最小值和最大值的平均值
        else:
            k1 = s1
            k2 = s2
            kj = np.mean([k1, k2])
        kjz2 = [[], []]
        for j in kjz1:
            if j <= kj:
                kjz2[0].append(j)
            else:
                kjz2[1].append(j)
        s1 = np.mean(kjz2[0])
        s2 = np.mean(kjz2[1])
        if bj == 2:
            if s1 == k1 and s2 == k2:
                break
        bj = 2
    return kjz2

