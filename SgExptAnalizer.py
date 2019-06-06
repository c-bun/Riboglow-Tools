
from batch_SG_analysis import *
import seaborn as sn
from os import listdir
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

class SgExptAnalizer:
    def __init__(self, halo_threshold, loading_sn, halo_channel=2, rg_channel=1):
        self.d = {
            'condition':[],
            'ratio':[],
            'filename':[]
        }
        self.df = None
        self.halo_threshold = halo_threshold
        self.loading_sn = loading_sn
        self.halo_channel = halo_channel
        self.rg_channel = rg_channel
        
    def addCondition(self, path, condition, largeImage=False):
        all_ratios = []
        all_filenames = []
        if not largeImage:
            for filename in listdir(path):
                if filename == ".DS_Store": continue
                finder = SgFinder(
                    path+"/"+filename,
                    self.loading_sn,
                    haloTag_channel=self.halo_channel,
                    rG_channel=self.rg_channel
                )
                finder.median_filter(3)
                print(filename+":")
                finder.setHaloThreshold(threshold=self.halo_threshold)
                finder.dilate_and_ratio(iterations=5)
                all_ratios = all_ratios + finder.ratios
                all_filenames = all_filenames + [filename for x in finder.ratios]
        else:
            finder = SgFinder(
                    path,
                    self.loading_sn,
                    haloTag_channel=self.halo_channel,
                    rG_channel=self.rg_channel
                )
            finder.median_filter(3)
            print(filename+":")
            finder.setHaloThreshold(threshold=self.halo_threshold)
            finder.dilate_and_ratio(iterations=5)
            all_ratios = finder.ratios
            all_filenames = [filename for x in finder.ratios]
        
        for pt in all_ratios:
            self.d['condition'].append(condition)
            self.d['ratio'].append(pt)
        for name in all_filenames:
            self.d['filename'].append(name)
        
        self.df = pd.DataFrame(self.d)
        
        return self.df
    
    def boxplot(self, df=None):
        if df is None: df = self.df
        ax = sn.stripplot(data=df,y='ratio',x='condition',jitter=True, alpha=0.6)
        return sn.boxplot(data=df,y='ratio',x='condition',ax=ax,color='w',fliersize=0)

    def anova(self, y='ratio', x='condition', df=None):
        if df is None: df = self.df
        mod_string = '{} ~ {}'.format(y,x)
        mod = ols(mod_string,
                        data=df).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        return aov_table
        