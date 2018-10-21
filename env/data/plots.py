import pandas as pd
import numpy as np
import config as conf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
img_path = conf.IMAGE_DIR


def plot_prices(df):
    ax = df.plot(kind='line', legend=False, figsize=(8.5, 4))
    patches, labels = ax.get_legend_handles_labels()
    plt.legend(patches, labels, bbox_to_anchor=(1.00, 0.8), loc=2, borderaxespad=0.)
    plt.xlabel('Datum')
    plt.ylabel('Preis in USD')
    plt.subplots_adjust(left=0.09, top=0.98, bottom=0.04)
    plt.savefig(img_path + '/asset_prices.png', dpi=96)
    plt.show()


def plot_returns(df):
    ax = df.plot(kind='line', legend=False, figsize=(8.5,4))
    patches, labels = ax.get_legend_handles_labels()
    plt.legend(patches, labels, bbox_to_anchor=(1.00, 0.8), loc=2, borderaxespad=0.)
    plt.xlabel('Datum')
    plt.ylabel('Kumulierte Rendite')
    plt.subplots_adjust(left=0.09, top=0.98)
    plt.savefig(img_path + '/asset_returns_line.png', dpi=96)
    plt.show()


def asset_plots(df1, df2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7), sharex='all')
    df1.plot(ax=ax1, legend=False, kind='line')
    df2.plot(ax=ax2, legend=False, kind='line')
    ax1.set_ylabel('Preis in USD')
    ax2.set_ylabel('Kumulierte Rendite')
    ax2.set_xlabel('Datum')
    patches, labels = ax2.get_legend_handles_labels()
    plt.legend(patches, labels, bbox_to_anchor=(1.00, 1.3), loc=2, borderaxespad=0.)
    fig.subplots_adjust(hspace=0.025, wspace=0.05, left=0.07,
                        top=0.98, bottom=0.05,right=0.88)
    fig.savefig(img_path + '/asset_data.png', dpi=96)
    fig.show()


def sums(returns):
    returns = np.array(returns)
    a = np.array([returns[1]])
    for i in range(1, returns.shape[0]):
        vals = a[i-1] + returns[i]
        a = np.append(a, [vals], axis=0)
    return a


def boxplot_return(df):
    plt.ylim(-0.04, 0.04)
    sns.boxplot(data=df, palette="RdBu")
    plt.xlabel('Anlage')
    plt.ylabel('Rendite')
    plt.savefig(img_path + '/asset_returns.png', dpi=96)
    plt.show()


def heatmap(df):
    sns.set_style('whitegrid')
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    fig.subplots_adjust(left=0.1, top=0.99, bottom=0.0125, right=0.99)
    plt.savefig(img_path + '/asset_heatmap.png', dpi=96)
    plt.show()


def asset_table(df):
    d = df.describe()
    d = d.drop(d.index[0])
    d = d.transpose()
    print(d)
    return d


if __name__ == '__main__':
    df = pd.read_csv('asset_price.csv', index_col=0)
    pct = df.pct_change(1)
    s1 = pct.cumsum()
    s2 = pd.DataFrame(sums(pct), columns=df.columns, index=df.index)
    #plot_prices(df)
    #plot_returns(s2)
    boxplot_return(pct)
    asset_table(pct)
    asset_plots(df, s1)
    heatmap(df)
