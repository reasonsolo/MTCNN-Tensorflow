import os
import sys

data_dir = '.'
#anno_file = os.path.join(data_dir, "anno.txt")

if __name__ == '__main__':
    net = sys.argv[1]

    with open(os.path.join(data_dir, '%s/pos_%s.txt' % (net, net)), 'r') as f:
        pos = f.readlines()

    with open(os.path.join(data_dir, '%s/neg_%s.txt' % (net, net)), 'r') as f:
        neg = f.readlines()

    with open(os.path.join(data_dir, '%s/part_%s.txt' % (net, net)), 'r') as f:
        part = f.readlines()

    with open(os.path.join(data_dir, '%s/landmark_%s.txt' % (net, net)), 'r') as f:
        landmark = f.readlines()

    dir_path = os.path.join(data_dir, 'imglists', net)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    #write all data
    with open(os.path.join(dir_path, "train_%s_landmark.txt" % (net)), "w") as f:
        print len(neg)
        print len(pos)
        print len(part)
        print len(landmark)
        for i in np.arange(len(pos)):
            f.write(pos[i])
        for i in np.arange(len(neg)):
            f.write(neg[i])
        for i in np.arange(len(part)):
            f.write(part[i])
        for i in np.arange(len(landmark)):
            f.write(landmark[i])
