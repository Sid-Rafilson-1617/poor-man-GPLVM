'''
more helper functions for my own paper writing
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import os,sys,copy,pdb

import seaborn as sns
import matplotlib

import misc

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# rcdict={'axes.labelsize':20,'axes.titlesize':20}
fs=10
rcdict={'font.size':fs,'axes.labelsize':fs,'axes.titlesize':fs,'xtick.labelsize':fs,'ytick.labelsize':fs,'legend.fontsize':fs}
matplotlib.rcParams.update(rcdict)
plt.rcParams['svg.fonttype'] = 'none'

def p_to_star(p_value):
    if 0.05 >= p_value > 0.01:
        star = "*"
    elif 0.01 >= p_value >0.001:
        star="**"
    elif 0.001 >= p_value >0.0001:
        star = "***"
    elif 0.0001 >= p_value:
        star = "****"
    else:
        star='n.s.'
    return star

def subplots_wrapper(nplots,return_axs=True,basewidth=6,baseheight=4,figsize=None,**kwargs):
    nrows = int(np.sqrt(nplots))
    ncols = int(nplots // nrows)
    if nplots%nrows !=0:
        ncols+=1
    if figsize is None:
        figsize=(ncols*basewidth,nrows*baseheight)
    if return_axs:
        
        fig,axs = plt.subplots(nrows,ncols,figsize=figsize,**kwargs)
        return fig,axs
    else:
        fig = plt.figure(figsize=figsize)
        return fig, nrows, ncols

def turn_off_spines(ax=None,to_turn=['top','right','left']):
    if ax is None:
        ax = plt.gca()
    for t in to_turn:
        ax.spines[t].set_visible(False)
    
    return ax

def plot_rasterplot(event_l=None,binary_mat=None,x_index=None,y_index=None,
                    fig=None,ax=None
                    ):
    '''
    event_l: list of list
    binary_mat: 2d array of spike or not
    x_index, y_index: convert the index like 0,1,2, to coordinates like time / position...
    '''
    
    if event_l is None:
        try:
            binary_mat_ = binary_mat.values # works for both pd and xr
        except:
            binary_mat_ = binary_mat
        ntrials = binary_mat_.shape[0]
        if x_index is None:
            event_l=[np.nonzero(binary_mat_[i,:]>0)[0] for i in range(ntrials)]    
        else:
            event_l=[x_index[binary_mat_[i,:]>0] for i in range(ntrials)]
    else:
        ntrials = len(event_l)
    if ax is None:
        fig,ax=plt.subplots()

    ax.eventplot(event_l,lineoffsets=1, linelengths=0.8, colors='black')
    yticks = np.arange(0,ntrials,int(ntrials//5))
    if y_index is not None:
        y_index_=y_index[yticks]
    else:
        y_index_ = yticks
    plt.yticks(ticks=yticks,labels=y_index_)
    return fig,ax,event_l
    



# def plot_using_two_axis(df,left_cols=None,right_cols=None,**kwargs):
#     kwargs_ = {}
#     kwargs_.update(kwargs)
#     if 'ax' not in kwargs_.keys():
#         fig,ax=plt.subplots()
#         kwargs_['ax'] = ax
#     if left_cols is None:
#         left_cols = df.columns
#     ax=df[left_cols].plot(**kwargs_)
#     if right_cols is not None:
#         ax_right = ax.twinx()
#         kwargs_['ax']=ax_right
#         ax=df[right_cols].plot(**kwargs_)
#     return ax 


def diverging_heatmap(X,quantile=0.99,**kwargs):
    vmax = np.quantile(np.abs(X),quantile)
    ax=sns.heatmap(X,cmap='vlag',center=0,vmin=-vmax,vmax=vmax,**kwargs)
    return ax
    
def plot_shuffle_data_dist_with_thresh(shuffle,data,bins=20,alpha=0.025,fig=None,ax=None,lw=4,plot_ci_high=True,plot_ci_low=False,figsize=(2,1.3)):
    thresh_high=np.quantile(shuffle,(1-alpha))
    percentile_high = (1-alpha) * 100
    # if plot_ci_low:
    thresh_low=np.quantile(shuffle,alpha)
    percentile_low = alpha * 100
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    ax.hist(shuffle,bins=bins,alpha=0.5)
    ax.axvline(data,label='data',linewidth=lw)
    if plot_ci_low:
        ax.axvline(thresh_low,label=f'{percentile_low:.02f} percentile',linestyle=':',linewidth=lw)
    if plot_ci_high:
        ax.axvline(thresh_high,label=f'{percentile_high:.02f} percentile',linestyle=':',linewidth=lw)
    ax.legend()
    return fig,ax

def mean_error_plot(df,xs=None,ci_scale=1.96,axis=0,fig=None,ax=None,alpha=0.5,label=None,sem=True,linestyle='-',c='C0'):
    '''
    plot line with shaded error region, mean + ci
    '''
    dfmean = df.mean(axis=axis)
    if sem:
        error = df.sem(axis=axis) * ci_scale
        dfup,dflow = dfmean + error, dfmean - error
    else:
        dfup = np.quantile(df.values,1-0.025,axis=axis)
        dflow = np.quantile(df.values,0.025,axis=axis)
    
    if ax is None:
        fig,ax=plt.subplots()
    if xs is None:
        xs = df.columns.astype(float)
    # pdb.set_trace()
    ax.fill_between(xs,dflow,dfup,alpha=alpha,color=c)
    ax.plot(xs,dfmean.values,label=label,linestyle=linestyle,color=c)
    return fig,ax

def mean_bootstraperror_lineplot(data_sample,data=None,xs=None,ci=0.95,fig=None,ax=None,**kwargs):
    '''
    given 1d data statistics and bootstrapped 1d statistics, plot line with errorbar
    data_sample: nsample x nfeature
    '''
    if ax is None:
        fig,ax=plt.subplots()
    if xs is None:
        xs = np.arange(data_sample.shape[1])
    q_high = (1-ci)/2 + ci
    y_high = np.quantile(data_sample,q_high,axis=0)
    q_low = (1-ci)/2 
    y_low = np.quantile(data_sample,q_low,axis=0)
    
    yerr = np.array([y_low,y_high])
    if data is None:
        data = np.mean(data_sample,axis=0)
    
    ax.errorbar(xs,data,yerr=yerr,**kwargs)
    return fig,ax

    
    

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
def plot_multipage_wrapper(plot_one_func,data_all,title_all=None,func_args=(),func_kwargs={}, nplots_per_page=20,fig_save_fn='fig_temp',fig_save_dir='./'):
    '''
    plot_one_func(data_one,*func_args,**func_kwargs)
    iterate this on all subplots
    
    data_all: list of data_one
    title_all: list or None

    '''
    plt.ioff()
    nplots = len(data_all)
    npages = int(np.ceil(nplots / nplots_per_page))
    plot_ind = 0
    fig_save_dir = misc.get_or_create_subdir('',fig_save_dir)
    fig_save_fn_full = os.path.join(fig_save_dir, f'{fig_save_fn}.pdf')


    with PdfPages(fig_save_fn_full) as pdf:
        for n in range(npages): 
            fig,axs = subplots_wrapper(nplots_per_page)
            for ii,ax in enumerate(axs.ravel()):
                try:
                    data_one = data_all[plot_ind]
                except IndexError:
                    break 
                _ =plot_one_func(data_one,fig=fig,ax=ax)
                if title_all is not None:
                    title = title_all[plot_ind]
                else:
                    title = plot_ind
                ax.set_title(title)
                plot_ind += 1
            pdb.set_trace()
            pdf.savefig(figure=fig,bbox_inches='tight')
            plt.close(fig=fig)
    
    print(f'{fig_save_fn_full} saved!',flush=True)


from matplotlib.animation import FuncAnimation,ArtistAnimation
from mpl_toolkits import mplot3d

def animate_3d_scatter(data,legend_handles=None,color_l=None,marker_l=None,fig=None,ax=None,ax_kws={},animate_kws={},dosave=False):
    '''
    data: 3 x T
    color_l: T,

    '''

    if color_l is None:
        color_l = ['C0'] * data.shape[1]
    if marker_l is None:
        marker_l = ['o'] * data.shape[1]
    if ax is None:
        fig=plt.figure()
        ax = fig.add_subplot(projection='3d')
    
    ax_kws_ = {}
    ax_kws['xlim'] = (data[0].min() - np.abs(data[0].min()) * 0.2, data[0].max() + np.abs(data[0].max())*0.2)
    ax_kws['ylim'] = (data[1].min() - np.abs(data[1].min()) * 0.2, data[1].max() + np.abs(data[1].max())*0.2)
    ax_kws['zlim'] = (data[2].min() - np.abs(data[2].min()) * 0.2, data[2].max() + np.abs(data[2].max())*0.2)
    ax_kws.update(ax_kws)

    def init():
        ax.set(**ax_kws)
        if legend_handles is not None:
            ax.legend(handles=legend_handles)
        return scat,

    def animate(frame,data,scat,color_l):
        scat._offsets3d = data[:,:frame]
        scat.set_color(color_l[:frame])
        # scat.set_marker(marker_l[:frame])
        return scat,
    
    scat = ax.scatter(data[0,0],data[1,0],data[2,0])
    
    kws = dict(blit=False,frames=data.shape[1],interval=10)
    kws.update(animate_kws)
    ani = FuncAnimation(fig,animate,init_func=init,fargs=(data,scat,color_l),**kws)
    plt.show()
    if dosave:
        pass
    return fig,ax,ani


def plot_pca3d(toplot,color=None,marker=None,fig=None,ax=None):
    if ax is None:
        fig=plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
    if color is None:
        color=np.array(['C0'] * toplot.shape[0])
    if marker is None:
        marker='.'
    if isinstance(marker,list) or isinstance(marker,np.ndarray):
        marker_type = pd.unique(marker)
        for mm in marker_type:
            ma = marker == mm
            ax.scatter3D(toplot[ma,0],toplot[ma,1],toplot[ma,2],color=color[ma],marker=mm)

    else:
        ax.scatter3D(toplot[:,0],toplot[:,1],toplot[:,2],color=color,marker=marker)
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')
    ax.set_zlabel('pc3')
    return fig,ax
    

# processing individual color/marker for scatter plot, according to features like maze section / speed
# could be made more general, for now seperate functions
import matplotlib.patches as mpatches
section_dict={'home':[0,15],'central':[15,74],'T':[74,111],'return side':[111,185],'return central':[185,222]}
def color_arms(lin,section_dict=None,color_l = None):
    if section_dict is None:
        section_dict={'home':[0,15],'central':[15,74],'T':[74,111],'return side':[111,185],'return central':[185,222]}
    if color_l is None:
        nsections = len(section_dict)
        color_l = ['C'+str(i) for i in range(nsections)]
    color_in_time = np.zeros_like(lin,dtype=object)
    legend_l = []
    for ii,(k,sec) in enumerate(section_dict.items()):
        if ii==0:
            mask = (lin >=sec[0]) & (lin<=sec[1])
        else:
            mask = (lin >sec[0]) & (lin<=sec[1])
        color_in_time[mask] = color_l[ii]
        legend =mpatches.Patch(color=color_l[ii], label=k)
        legend_l.append(legend)
        
    return color_in_time,legend_l,section_dict,color_l

from matplotlib.lines import Line2D
def marker_speed(speed,speed_range_dict=None,marker_l=None):
    
    if speed_range_dict is None:
        speed_range_dict = {'nan':np.nan,'stationary':[0,2],'low speed':[2,10],'high speed':[10,100]}
    if marker_l is None:
        marker_l_all = ['x','.','v','*','<','d','p','s'] # to be added or made more principled
        nsections = len(speed_range_dict)
        assert nsections <= len(marker_l_all)
        marker_l = marker_l_all[:nsections]
    marker_in_time = np.zeros_like(speed,dtype=object)
    legend_l = []
    for ii,(k,sec) in enumerate(speed_range_dict.items()):
        if isinstance(sec,list):
            if ii ==0:
                mask = (speed >=sec[0]) & (speed<=sec[1])
            else:
                mask = (speed >sec[0]) & (speed<=sec[1])
        else:
            mask = np.isnan(speed)
        marker_in_time[mask] = marker_l[ii]
#         legend =mpatches.Patch(marker=marker_l[ii], label=k)
#         legend =plt.scatter([],[],marker=marker_l[ii], label=k,color='k')
        legend=Line2D([0], [0], marker=marker_l[ii], color='k', label=k)
        legend_l.append(legend)
    return marker_in_time,legend_l,speed_range_dict,marker_l
        
def prep_hue_scatter(spks_onetrial_,speed_key='v'):
    '''
    spks_onetrial_: df; actually only need ['lin',speed_key]
    '''
    lin = spks_onetrial_['lin'].values
    color_in_time,legend_l_color,section_dict,color_l = color_arms(lin)

    speed = spks_onetrial_['v'].abs().values
    marker_in_time,legend_l_marker,speed_range_dict,marker_l = marker_speed(speed,speed_range_dict=None,marker_l=None)

    legend_l = legend_l_color+legend_l_marker
    
    return color_in_time,marker_in_time,legend_l,legend_l_color,legend_l_marker


# plotting vertical lines on a 1d plot vs position
def plot_vlines(lin_val_left,fig=None,ax=None):
    if ax is None:
        fig,ax= plt.subplots()
    for l in lin_val_left:
        ax.axvline(l,color='k',linestyle=':')
    return fig,ax

def plot_section_markers(labels=None,bounds=None,fig=None,ax=None,rotation=20):
    '''
    on any 1-d plot using lin / lin_binned as x, plot vertical lines marking landmarks
    '''
    if bounds is None:
        bounds = np.array([0,15,74,111,185,222])
        bounds = bounds / 2.2
    if labels is None:
        labels = ['home','central','T','return side','return central']
    minor_ticks = (bounds[:-1] + bounds[1:])/2

    if ax is None:
        fig,ax=plt.subplots(figsize=(10,6))
    ax.set_xticks([])
    ax.set_xticklabels('')

    ylims=ax.get_ylim() # because of this, this func should be the last step
    for xx in bounds[1:-1]:
        ax.vlines(xx,ylims[0],ylims[1],color='k',linestyle=':')
    ax.set_xticks(minor_ticks,minor=True)
    ax.set_xticklabels(labels,minor=True,rotation=rotation)
    return fig,ax


# like in hmm, shade time points according to the hidden state
def plot_shades(labels,fig=None,ax=None):
    '''
    # like in hmm, shade time points according to the hidden state
    only work for imshow now, but can adjust 
    '''
    if ax is None:
        fig,ax=plt.subplots()
    clust_l = np.unique(labels)
    labels_extended = np.insert(labels,0,-1)
    labels_extended = np.append(labels_extended,-1)
    for ii,c in enumerate(clust_l):
        true_segments = labels_extended == c
        true_segments_diff = np.diff(true_segments.astype(int))
        seg_starts=np.nonzero(true_segments_diff==1)[0] 
        seg_ends = np.nonzero(true_segments_diff==-1)[0] 
        for (st,ed) in zip(seg_starts,seg_ends):
            ax.fill_between(np.array([st,ed]),0,1,color=f'C{ii}', alpha=0.3, transform=ax.get_xaxis_transform())
    return fig,ax
    


### barebone version of ratemap with field bounds
def ratemap_one_raw(data,trial=None,field_bound=None,fig=None,ax=None,line_kws={},title=None,heatmap_kws={}):

    line_kws_ = {'linewidth':2,'linestyle':':'}
    line_kws_.update(line_kws)
    # heatmap_kws_ = {'cmap':'Greys','cbar':True}
    heatmap_kws_ = dict(vmin=0,vmax=None,vmax_quantile=0.99,cmap='viridis',xlabel='Position',ylabel='',cbar=True,cbar_ax=None)
    heatmap_kws_.update(heatmap_kws)
    if ax is None:
        fig,ax=plt.subplots(figsize=(3,2))
    # ax=sns.heatmap(data,ax=ax,**heatmap_kws_)

    fig,ax=heatmap(data,fig=fig,ax=ax,**heatmap_kws_)
    
    hcolor = 'pink'
    vcolor = 'red'
    if trial is not None:
        if hasattr(trial, '__iter__'):
            for tr in trial:
                if tr is not None:
                    ax.hlines(tr,*ax.get_xlim(),color=hcolor,**line_kws_)
        else:
            ax.hlines(trial,*ax.get_xlim(),color=hcolor,**line_kws_)
    if field_bound is not None:
        field_st,field_end = field_bound        
        ax.vlines(field_st,*ax.get_ylim(),color=vcolor,**line_kws_)
        ax.vlines(field_end,*ax.get_ylim(),color=vcolor,**line_kws_)
    if title is not None:
        ax.set_title(title)
    return fig,ax

def plot_field_bound(field_st,field_end,ax=None,fig=None,c='C1',line_kws={}):
    line_kws_ = {'linewidth':3,'linestyle':':'}
    line_kws_.update(line_kws)
    if ax is None:
        fig,ax=plt.subplots()
    ax.vlines(field_st,*ax.get_ylim(),color=c,**line_kws_)
    ax.vlines(field_end,*ax.get_ylim(),color=c,**line_kws_)
    return fig,ax

def plot_switch_trial(trial,xlim=None,c='C0',fig=None,ax=None,line_kws={}):
    line_kws_ = {'linewidth':3,'linestyle':':'}
    line_kws_.update(line_kws)
    if ax is None:
        fig,ax=plt.subplots()
    if xlim is None:
        xlim = ax.get_xlim()
    if trial is not None:
        if hasattr(trial, '__iter__'):
            for tr in trial:
                if tr is not None:
                    ax.hlines(tr,*xlim,color=c,**line_kws_)
        else:
            ax.hlines(trial,*xlim,color=c,**line_kws_)
    return fig,ax



# colorbar
def plot_colorbar(**kwargs):
    # Create a new figure
    figsize = kwargs.get('figsize',(0.2,4))
    fig, ax = plt.subplots(figsize=figsize)

    # Define the color map
    # cmap = mpl.cm.Greys
    cmap = mpl.cm.viridis
    cmap = kwargs.get('cmap',cmap)

    # Define the normalization
    vmin=kwargs.get('vmin',0)
    vmax=kwargs.get('vmax',1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Create the colorbar
    orientation= kwargs.get('orientation','vertical')
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=orientation)
    return fig,ax,cb
    
def prep_color(colors,cmap_str='jet',color_discrete = False):
    cmap = plt.get_cmap(cmap_str)
    if color_discrete:
        levels = MaxNLocator(nbins=int(cmax-cmin+1)).tick_values(cmin,cmax+1)
        norm = BoundaryNorm(levels,ncolors=cmap.N,clip=True)
    else:
        cmin,cmax=np.min(colors),np.max(colors)
        norm = plt.Normalize(cmin, cmax)
    c = cmap(norm(colors))
    return c

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def plot_cdf_and_ks_test(sample1, sample2, alpha=0.05,fig=None,ax=None,label1='sample1',label2='sample2',xlabel='Value',title=None,
                        c_l  = ['C0','C1'],linestyle_l=['-','-'],alternative='two-sided',do_legend=True,
                        bins='auto',
                    ):
    _,bin_edges = np.histogram(np.concatenate([np.array(sample1),np.array(sample2)]),bins=bins,density=True)
    # Compute CDF for sample1
    # hist1, bin_edges1 = np.histogram(sample1, bins='auto', density=True)
    hist1, bin_edges1 = np.histogram(sample1, bins=bin_edges, density=True)
    cdf1 = np.cumsum(hist1) / np.sum(hist1)
    
    # Compute CDF for sample2
    # hist2, bin_edges2 = np.histogram(sample2, bins='auto', density=True)
    hist2, bin_edges2 = np.histogram(sample2, bins=bin_edges, density=True)
    cdf2 = np.cumsum(hist2) / np.sum(hist2)
    if ax is None:
        fig,ax=plt.subplots()
    # Plot CDFs
    ax.plot(bin_edges1[1:], cdf1, label=label1,c=c_l[0],linestyle=linestyle_l[0])
    ax.plot(bin_edges2[1:], cdf2, label=label2,c=c_l[1],linestyle=linestyle_l[1])
    
    # KS test
    ks_stat, p_value = ks_2samp(sample1, sample2,alternative=alternative)
    
    common_bins = np.union1d(bin_edges1, bin_edges2)
    cdf1_interp = np.interp(common_bins, bin_edges1[1:], cdf1)
    cdf2_interp = np.interp(common_bins, bin_edges2[1:], cdf2)
    max_diff_idx = np.argmax(np.abs(cdf1_interp - cdf2_interp))
    max_diff_x = common_bins[max_diff_idx]
    max_diff_y = max(cdf1_interp[max_diff_idx], cdf2_interp[max_diff_idx])
    
    
    if p_value < alpha:
        title_ = f"KS Test: p-value = {p_value:.3f} *\nstat={ks_stat:.3f}"
    else:
        title_ = f"KS Test: p-value = {p_value:.3f}\nstat={ks_stat:.3f}"
    
    if title is None:
        title=title_
    ax.set_title(title)
    
    if p_value < alpha:
        if 0.05 >= p_value > 0.01:
            star = "*"
        elif 0.01 >= p_value >0.001:
            star="**"
        elif 0.001 >= p_value >0.0001:
            star = "***"
        elif 0.0001 >= p_value:
            star = "****"


        # ax.annotate(f"* p={p_value:.3f}", (max_diff_x, max_diff_y + 0.05), 
        #              ha='center', va='bottom', color='k')
        ax.annotate(f"{star}", (max_diff_x, max_diff_y + 0.05), 
                     ha='center', va='bottom', color='k')
    else:
        ax.annotate(f"n.s.", (max_diff_x, max_diff_y + 0.05), 
                     ha='center', va='bottom', color='k')
    if do_legend:
        ax.legend(bbox_to_anchor=[1.05,1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ax.grid(False)
    sns.despine()
#     plt.show()

    return fig,ax

# lineplot with errorbar, given a wide form df: nsamples x ntrials
from matplotlib.ticker import MaxNLocator
def mean_across_row_vs_col_with_err(df,value_name='value',var_name='Trial',integer_x=True,cols_reset=None,fig=None,ax=None,plot_type='line',**kwargs):
    df_ = copy.copy(df)
    cols = df.columns
    if cols_reset is None:
        df_.columns = np.arange(df_.shape[1])
    else:
        df_.columns = cols_reset
    df_ = df_.melt(var_name=var_name,value_name=value_name)
    if ax is None:
        fig,ax=plt.subplots()
    if plot_type=='line':
        sns.lineplot(data=df_,x=var_name,y=value_name,ax=ax,**kwargs)
    elif plot_type=='bar':
        sns.barplot(data=df_,x=var_name,y=value_name,ax=ax,**kwargs)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig,ax
    

# mark the days
def plot_day_on_heatmap(df,axis=0,level=0,vline=False,hline=True,ax=None,fig=None,color='C0'):
    if axis==0:
        day_l=df.index.get_level_values(level=level)
    else:
        day_l=df.columns.get_level_values(level=level)
    day_change = np.nonzero(np.diff(day_l))[0]+1
    if ax is None:
        fig,ax=plt.subplots()
    for dd in day_change:
        if hline:
            ax.axhline(dd,color=color,linestyle=':')
        if vline:
            ax.axvline(dd,color=color,linestyle=':')
    return fig,ax

# simple heatmap 
from matplotlib.ticker import MaxNLocator,MultipleLocator
def heatmap(df,fig=None,ax=None,vmin=0,vmax=None,vmax_quantile=0.99,cmap='viridis',xlabel='Position',ylabel='',cbar=True,cbar_ax=None):
    if ax is None:
        fig,ax=plt.subplots()
    xx=df.values.flatten()
    xx=xx[~np.isnan(xx)]
    if vmax is None:
        vmax = np.nanquantile(xx,vmax_quantile)
    
    sns.heatmap(df, cmap=cmap,vmax=vmax,vmin=vmin,ax=ax,cbar_ax=cbar_ax,cbar=cbar)
    ax.set(xlabel=xlabel,ylabel=ylabel)

    locator=MaxNLocator(nbins=5,integer=True)
    ax.xaxis.set_major_locator(locator)
    # Retrieve the positions of the current ticks
    x_ticks = ax.get_xticks()
    # Map tick positions to integer indices (assuming linear relationship)
    x_tick_indices = np.round(x_ticks).astype(int)
    # Ensure indices are within the bounds of the DataFrame's columns
    x_tick_indices = x_tick_indices[(x_tick_indices >= 0) & (x_tick_indices < len(df.columns))]
    # Retrieve corresponding column names
    x_tick_labels = df.columns[x_tick_indices]
    # Set tick labels
    ax.set_xticklabels(x_tick_labels, rotation=0)
    
    
    return fig,ax

def add_field_bounds(field_bounds,df=None,by_day=False,by_previous_day=False,fig=None,ax=None,**kwargs):
    kwargs_ = {'c':'C1','linestyle':':','linewidth':3}
    kwargs_.update(kwargs)
    if ax is None:
        fig,ax=plt.subplots()
    if df is not None:
        day_l = df.index.get_level_values(0)
        plot_height=df.shape[0]
    for i,row in field_bounds.iterrows():
        if by_day:
            day = i[0]
            ma = day_l == day
            change = np.diff(ma,append=False,prepend=False)
            ymin,ymax = np.nonzero(change)[0]
            ymin = 1-ymin/plot_height
            ymax = 1-ymax / plot_height
            
        elif by_previous_day: # if elif else!!!! not if if else
            day = i[0]
            ma = day_l <= day
            change = np.diff(ma,append=False,prepend=False)
            ymin,ymax = np.nonzero(change)[0]
            ymin = 1-ymin/plot_height
            ymax = 1-ymax / plot_height

        else:
            ymin=0
            ymax=1


        ax.axvline(row['start'],ymin,ymax,**kwargs_)
        ax.axvline(row['end'],ymin,ymax,**kwargs_)
    
    return fig,ax

# import statannotations
# from statannotations.Annotator import Annotator

# def paired_line_with_box(df,x,y,fig=None,ax=None,dotest=True,rotation=45,palette="coolwarm",**line_kws):
#     '''
#     df columns can only contain the x values
#     '''
#     line_kws_={'marker':'o','color':'gray','alpha':0.2,'ms':0.5}
#     line_kws_.update(line_kws)
#     if ax is None:
#         fig,ax=plt.subplots(figsize=(2,2))
#     for index, row in df.iterrows():
#         ax.plot(np.arange(2),[row[x],row[y]],**line_kws_)
#     boxdf=df.melt()
#     col_name=df.columns.name
#     if col_name is None:
#         col_name='variable'
#     # sns.boxplot(data=boxdf,x=col_name,y='value',ax=ax,palette="coolwarm", width=0.3)
#     sns.boxplot(data=df,ax=ax,palette=palette, width=0.3,order=[x,y])
#     sns.despine()
#     # ax.tick_params(axis='x',labelsize=15)
#     ax.set_xticklabels(ax.get_xticklabels(),rotation=rotation)
#     if dotest:
#         annotator = Annotator(ax, pairs=[df.columns], x=col_name, y='value',data=df.melt())
#         # annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
#         annotator.configure(test='Wilcoxon', text_format='star', loc='outside')
#         annotator.apply_and_annotate()
#     return fig,ax

def save_given_name(fig,figfn,figdir='',dpi='figure'):
    figfn_clean = figfn.replace('.','dot')
    for fmt in ['png','svg']:
        figfn_full = f'{figfn_clean}.{fmt}'
        fig.savefig(os.path.join(figdir,figfn_full),bbox_inches='tight',dpi=dpi)
        
def box_strip_plot(df,x1,x2,hue=None,fig=None,ax=None,line_kws={},do_logy=False):
    line_kws_ = {'c':'Grey','alpha':0.3}
    line_kws_.update(line_kws)
    xs = np.arange(2)+1
    if ax is None:
        fig,ax=plt.subplots()
    for i,row in df.iterrows():
        ax.plot(xs,row[[x1,x2]].values,**line_kws_)
    ax.boxplot(df[[x1,x2]].values)
    if do_logy:
        ax.set_yscale('log')
    sns.despine()
    ax.set_xticklabels([x1,x2])
    return fig,ax
    
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# just plot legend

def plot_legend(labels, colors=None, handle_types=None,vertical=True,frameon=False,figsize=(0.6,0.4)):
    """
    Creates a figure with a standalone legend.

    Args:
        labels (list): A list of label strings.
        colors (list): A list of color strings.
        handle_types (list): A list of handle types e.g., "line" or "patch".
    """
    if colors is None:
        colors = [f'C{i}' for i in range(len(labels))]
    if handle_types is None:
        handle_types = ['patch'] * len(labels)
    
    # Create custom handles
    handles = []
    for color, handle_type in zip(colors, handle_types):
        if handle_type == "line":
            handles.append(Line2D([0], [0], color=color, lw=2))
        elif handle_type == "patch":
            handles.append(Patch(facecolor=color))
        # Add more handle types as needed with elif blocks

    # Create a new figure for the legend
    fig,ax = plt.subplots(figsize=figsize)  # Adjust the size as needed
    if vertical:
        ncol=1
    else:
        ncol = len(labels)
            
    fig.legend(handles=handles, labels=labels, loc='center', ncol=ncol,frameon=frameon)
    ax.axis('off')
    
    return fig,ax


def median_plot(**kwargs):
    kwargs_hide=dict(
            boxprops=dict(visible=False),
            whiskerprops=dict(visible=False),
            capprops=dict(visible=False),
            flierprops=dict(visible=False)
            )
    kwargs.update(kwargs_hide)
    g=sns.boxplot(**kwargs)
    return g

def star_map(r):
    if r>0.05:
        star ='n.s.'
    elif 0.01<r<=0.05:
        star = '*'
    elif 0.001<r<=0.01:
        star='**'
    elif 0.0001<r<=0.001:
        star = '***'
    elif r<=0.0001:
        star='****'
    return star

def consecutive_wilcoxon_test(df):
    cols = df.columns
    res_all = {}
    for i in range(1,len(cols)):
        diff = df[cols[i]] - df[cols[i-1]]
        res=scipy.stats.wilcoxon(diff.dropna(axis=0))
        res_all[(cols[i-1],cols[i])] = res
    return res_all


# for small plots, only keep two ticks 
def set_two_ticks(axis, ylim=None,do_int=True):
    if ylim is None:
        ylim = axis.get_ylim()
    if do_int:
        ylim = [int(k) for k in ylim]
    axis.set_yticks([ylim[0], ylim[1]])
    return axis

# Helper function to apply consistent color to axis components
def apply_axis_color(axis, color,spine_side='left'):
    axis.spines[spine_side].set_color(color)  # Spine color
    axis.tick_params(axis='y', colors=color)  # Tick and tick label color
    axis.yaxis.label.set_color(color)  # Y-axis label color

import matplotlib.ticker as ticker
# Integer tick formatter
int_formatter = ticker.FuncFormatter(lambda x, _: f"{int(x)}")


import matplotlib
def plot_angle(theta, ds_fac=10,xs=None,y_coord=1,fig=None,ax=None,arrow_size=10,width=0.003,c='C2'):
    try:
        theta_val=theta.values
    except:
        theta_val =theta
    if xs is None:
        try:
            xs = theta.index

        except:
            xs = np.arange(len(theta))
            
    
    xs_ds = xs[::ds_fac]
    theta_ds = theta_val[::ds_fac]
    
    dx_l =  np.cos(theta_ds) * arrow_size
    dy_l =  np.sin(theta_ds) * arrow_size
    
    if ax is None:
        fig,ax=plt.subplots()
    
    
    x_pos = xs_ds
    y_pos = y_coord * np.ones_like(x_pos)
    ax.quiver(x_pos,y_pos,dx_l,dy_l, fc=c, ec=c,width=width
             )
    

    return fig,ax
    