#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import sys
from IPython import embed
from pylab import rcParams
from matplotlib.ticker import FormatStrFormatter
import argparse


def read_data(filep, check_names=None, check_methods=None):
    data_raw = [f.strip().split(' ',1) for f in open(filep).readlines()]
    names = data_raw[0][1].split()
    names = [n.replace("_"," ") for n in names]
    methods = [l[0] for l in data_raw[1:]]
    if check_names:
        assert np.all(names==check_names)    
    if check_methods:
        assert np.all(methods==check_methods)    
    return dict([(l[0], np.array(l[1].split(),dtype=float)) for l in data_raw[1:]]), names, methods
    

parser = argparse.ArgumentParser()
parser.add_argument('--mins',         help='Table with min values to be added to the bars', default=None)
parser.add_argument('--counts',       help='Table with counts to be printed under the xlabels', default=None)
parser.add_argument('--ylabel',       help='Name for ylabel', default=None)
parser.add_argument('--ymax',         help='Max for yaxis', default="from_data")
parser.add_argument('--colors',       help='List of colors to use', default=None)
parser.add_argument('--title',        help='Plot title', default=None)
parser.add_argument('--no_legend',    help='Do not include the legend', action='store_true')
parser.add_argument('table',          help='Table with results for each system (rows) for each group (columns). The first line should have the names of the groups. The first column should have the names of the systems.')
parser.add_argument('output',         help='Name for output pdf with plots.')

opt = parser.parse_args()

fsize1 = 21
fsize2 = 18
fsize3 = 25
rcParams['figure.figsize'] = 10,4.5
rcParams['xtick.labelsize'] = fsize1
rcParams['ytick.labelsize'] = fsize2
rcParams.update({'figure.autolayout': True})

font1 = FontProperties()
font2 = FontProperties()
font3 = FontProperties()
font1.set_size(fsize1)
font2.set_size(fsize2)
font3.set_size(fsize3)

if opt.colors:
    colors = opt.colors.split(",")
else:
    colors = ['cornflowerblue', 'darksalmon', 'palegoldenrod', 'blue']

data, names, methods = read_data(opt.table)

if opt.counts: 
    data_cnt, _, _ = read_data(opt.counts, check_names=names, check_methods=methods)

if opt.mins:
    data_min, _, _ = read_data(opt.mins,   check_names=names, check_methods=methods)

# Params for the bars
pos = np.array(np.arange(len(names)),dtype=float)
nmethods = len(methods)
width = 1.0/(nmethods+1)

xmin = min(pos)-width
xmax = max(pos)+width*(nmethods)

ymin = 0.0
if opt.ymax == "from_data":
    # Define the ymax as the min between:
    # 1) 2.00 from the max value of the last system (assumed to be the best)
    # 2) 1.05 from the max value of the first system (assumed to be the worst)
    ymax = min(np.max(data[methods[-1]])*1.7, np.max(data[methods[0]])*1.01)
else:
    ymax = float(opt.ymax)

with PdfPages(opt.output) as pdf:

    plt.figure()
    ax = plt.gca()

    for j, method in enumerate(methods):

        perf = data[method]
        method_name = method.replace("_", " ")

        plt.bar(pos+j*width, perf, width, alpha=1, color=colors[j], label=method_name, edgecolor="dimgrey")
        if opt.mins:
            plt.bar(pos+j*width, data_min[method], width, alpha=1, color=colors[j], edgecolor="black", linewidth=2)

        # Print the performance values on top of the bars and, optionally, the counts under the x-axis
        for k in np.arange(len(names)):
            f = "%.3f"%perf[k]
            ax.text(pos[k]+j*width, (ymax-ymin)*0.01, f, ha='center', va='bottom', fontproperties=font2, color='k', rotation=90)

            if j == 0 and opt.counts:
                ax.text(pos[k]+(nmethods-1)*width/2.0, ymin-(ymax-ymin)*0.10, int(data_counts["POS"][k]), ha='center', va='bottom', fontproperties=font2)
                ax.text(pos[k]+(nmethods-1)*width/2.0, ymin-(ymax-ymin)*0.15, int(data_counts["NEG"][k]), ha='center', va='bottom', fontproperties=font2)

            
    # Set the position of the x ticks
    ax.set_xticks(pos+width*(nmethods-1)/2)
        
    # Set the labels for the x ticks
    ax.set_xticklabels(names)
        
    # Setting the x-axis and y-axis limits
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.gcf().set_size_inches(len(names)*(1+len(methods))*0.5,8)

    if opt.ylabel:
        ax.set_ylabel(opt.ylabel, fontsize=fsize1)
    
    if not opt.no_legend:
        ax.legend(fontsize=fsize1)

    if opt.title:
        ax.set_title(opt.title, fontsize=fsize1)
#        ax.text((xmax+xmin)/2, ymax-(ymax-ymin)*0.07, opt.title, ha='center', va='bottom', fontproperties=font3, bbox=dict(facecolor='white', edgecolor='grey', alpha=0.7))
    
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()


