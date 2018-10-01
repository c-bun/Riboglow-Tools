
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sn
import numpy as np
from os import listdir

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

class Profiler:
    def __init__(self, path, positive_control_column=1, experimental_column=2):
        self.path = path
        self.positive_control_column = positive_control_column
        self.experimental_column = experimental_column
        self.files = []
        
    def parse_filename(self, name):
        tosplit = name.replace('.','_')
        components = tosplit.split('_')
        d = {
            'full_name' : name,
            'date' : components[0],
            'condition' : components[1]+components[2],
            'image_num' : components[3],
            'profile_num' : components[6]
        }
        return d
    
    def load_files(self, parser=parse_filename):
        for file in listdir(self.path):
            if file[-3:] == 'csv': self.files.append(parser(self,file))

    def load_trace(self, filepath):
        return np.loadtxt(filepath, delimiter=',',skiprows=1)

    def min_edges_trace(self, trace, n):
        return np.mean(np.hstack([trace[:n,self.experimental_column],trace[-n:,self.experimental_column]]))

    def remove_bkgd(self, trace):
        m = trace.min(axis=0)
        return trace-m

    def anova(self, y='ratio', x='condition', df=None):
        if df is None: df = self.df
        mod_string = '{} ~ {}'.format(y,x)
        mod = ols(mod_string,
                        data=df).fit()

        aov_table = sm.stats.anova_lm(mod, typ=2)
        return aov_table

    def tukey_hsd(self, y='ratio', x='condition', df=None):
        if df is None: df = self.df
        mc = MultiComparison(df[y], df[x])
        result = mc.tukeyhsd()
        return result.summary()

    # First, caluculate all the background averages
    def calculate_backgrounds(self):
        backgrounds = {}
        for file in self.files:
            if file['profile_num'] != '0':
                continue
            # find the average of each channel
            trace = self.load_trace(self.path+file['full_name'])
            slice1 = np.mean(trace[:,1])
            slice2 = np.mean(trace[:,2])
            slice3 = np.mean(trace[:,3])
            backgrounds[file['condition']+file['image_num']] = (slice1, slice2, slice3) # this is still a bit dirty, but it works for now?
        self.backgrounds = backgrounds

    def parse_granules(self):
        d = {'condition':[],
        #    'values':[], # for debugging
             'background':[],
             'filename':[], # for debugging
            'avg_min':[],
            'max':[],
            'ratio':[]}
        for file in self.files:
            if file['profile_num'] == '0': continue
            bkgd = self.backgrounds[file['condition']+file['image_num']] # This is bad. Should have background info merged into file info?
            d['background'].append(bkgd[self.experimental_column-1])
            d['filename'].append(file['full_name'])
            d['condition'].append(file['condition'])
            trace = self.load_trace(self.path+file['full_name'])
            avg_min = self.min_edges_trace(trace, 4) - bkgd[self.experimental_column-1]
            if avg_min < 1: avg_min = 1 # To eliminate wierd behaviour when taking a ratio with a small number in the denominator
            d['avg_min'].append(avg_min)
            peak = self.max_range(trace[:,self.positive_control_column],5)
            max_val = np.max(trace[peak,self.experimental_column]) - bkgd[self.experimental_column-1]
            d['max'].append(max_val)
            d['ratio'].append(max_val/avg_min)
        self.df = pd.DataFrame(d)
        return self.df

    def max_range(self, trace, width):
        max_index = np.argmax(trace)
        lower_bound = max_index-(width//2)
        if lower_bound < 0: lower_bound = 0
        upper_bound = lower_bound+width
        if len(trace) <= upper_bound: upper_bound = len(trace)-1
        return range(lower_bound, upper_bound)
    
    def boxplot(self):
        ax = sn.stripplot(data=self.df,y='ratio',x='condition',jitter=True)
        return sn.boxplot(data=self.df,y='ratio',x='condition',ax=ax,color='w',fliersize=0)