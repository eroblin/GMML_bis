import matplotlib.pyplot as plt
from matplotlib import collections as mc
plt.style.use('ggplot')


def new_segment(p1, p2):
    return mc.LineCollection([[p1, p2]], colors='#deebf7', linewidths=1.5)


def plot_embeddings(pdata, pdict, pembs, name, nb_epoch):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.patch.set_facecolor('white')
    ax.add_artist(plt.Circle((0, 0), 1., color='#1c9099', fill=False))
    for w,i in pdict.items():
        c0, c1 = pembs[i]
        ax.plot(c0, c1, 'o', color='#8A0808')
        ax.text(c0+.01, c1+.01, w, color='#08088A')
    for links in pdata:
        (point_1_x, point_1_y) = pembs[pdict[links[0]]]
        (point_2_x, point_2_y) = pembs[pdict[links[1]]]
        line = new_segment([point_1_x, point_1_y], [point_2_x, point_2_y])
        ax.add_collection(line)
    fig.savefig('outputs/%s_%s.png' % (name, nb_epoch), dpi=fig.dpi)