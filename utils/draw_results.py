import mne
from mne.viz.utils import _prepare_joint_axes, _connection_line
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from pylab import mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_topomap_joint(evoked, vmask, vmin, vmax, time_points, time_vline=[0], 
                        cmap='turbo', units='', figsize=(5, 2.5),
                        major_tick=0.1, minor_tick=0.01, vmask_delete=0,
                        cbar_ticks=[], marker_size=6) :
    
    font = {'family' : ['Arial', 'Times New Roman', 'serif' ], 
            'color'  : 'k',
            'weight' : 'normal',
            'size'   : 9,
            }
    fonts = {'family': ['Arial', 'Times New Roman', 'serif' ], 
            'color'  : 'k',
            'weight' : 'normal',
            'size'   : 8,
            }

    mpl.rcParams["axes.unicode_minus"] = False    

    # prepare axes
    fig, main_ax, map_axs = _prepare_joint_axes(len(time_points), figsize=figsize) 

    for row, ma in enumerate(vmask):
        start = 0
        end = 0
        state = False
        for ind, m in enumerate(ma):
            if m==True and state==False:
                state = True
                start = ind
            if m==False and state==True:
                state = False
                end = ind
                
                if (end-start)<vmask_delete:
                    for i in range(start, end):
                        vmask[row, i] = False

    # plot effect map
    subfig = evoked.plot_image( mask = vmask, 
                                mask_alpha = 1.0,
                                mask_cmap = cmap,
                                cmap = cmap,
                                show_names = True,
                                colorbar = True,
                                axes = main_ax,
                                show = False,
                                clim = dict(eeg=[vmin, vmax]), 
                                scalings = dict(eeg=1),
                                units = units,
                            )
    main_ax.set_title('')
    vmin, vmax = main_ax.images[0].get_clim() # remember color limits

    main_ax.xaxis.set_major_locator(MultipleLocator(major_tick))
    main_ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
    
    main_ax.spines['top'].set_linewidth(0.5)
    main_ax.spines['right'].set_linewidth(0.5)
    main_ax.spines['bottom'].set_linewidth(0.5)
    main_ax.spines['left'].set_linewidth(0.5)
    main_ax.tick_params(which='both', bottom=True, top=False, left=True, right=False,
            labelbottom=True, labelleft=True, direction='out',width=0.5)
    
    main_ax.set_yticks([0,20,40,60], [evoked.ch_names[0],evoked.ch_names[20],evoked.ch_names[40],evoked.ch_names[60]], fontdict=fonts)
    main_ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], ['0.0','','0.2','','0.4','','0.6','','0.8','','1.0'], fontdict=fonts);
        
    main_ax.set_xlabel('Time (s)', fontdict=font)
    main_ax.set_ylabel('Channels', fontdict=font) 
    
    cbar_ax = subfig.axes[-1]
    for a in subfig.axes:
        if isinstance(a, matplotlib.colorbar.Colorbar):
            cbar_ax = a

    tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数
    cbar_ax.locator = tick_locator
    
    if len(cbar_ticks) >0 :
        cbar_ax.set_yticks(cbar_ticks,cbar_ticks)
    cbar_ax.tick_params(labelsize=8)
    cbar_label = cbar_ax.get_xticklabels() 
    [lab.set_fontname('Arial') for lab in cbar_label]   
    
    evoked.plot_topomap( times = time_points, 
                         time_format = "%3.2f s", 
                         axes = map_axs, 
                         cmap = cmap,
                         mask = vmask, 
                         mask_params = {'markersize':marker_size},
                         vlim=(vmin, vmax),  # enforce same color limits as main
                         scalings=dict(eeg=1),
                         colorbar = False,
                         show = False,                         
                         )

    # draw connecting lines
    lines = [_connection_line(timepoint, fig, main_ax, map_ax_)
                 for timepoint, map_ax_ in zip(time_points, map_axs)]
    for line in lines:
        fig.lines.append(line)
    for timepoint in time_points:
        main_ax.axvline(timepoint, color = 'grey', linestyle = '-',
                          linewidth = 0.75, alpha = .7)

    if len(time_vline) == 0:
        for time_v in [1.5, 3.0]:
            main_ax.axvline(time_v, color = 'gray', linestyle = '--',
                                linewidth = 1.0, alpha = .9)
        for time_v in [1.0, 2.5]:
            main_ax.axvline(time_v, color = 'gray', linestyle = '--',
                                linewidth = 1.0, alpha = .9)
    elif time_vline==[0]:
        pass
    else:
        for time_v in time_vline:
            main_ax.axvline(time_v, color = 'gray', linestyle = '--',
                                linewidth = 1.0, alpha = .9)
  
    # show final plot
    plt.show()



def draw_tf_diff_clusters(F_stats, pow_list, epo_tfr, title='', cbunit='', 
                          vmin=0, vmax=1, cmap='turbo', figsize=(4.5, 2.5), cbar_ticks=[], markersize=5):
    
    font = {'family' : ['Arial', 'Times New Roman', 'serif' ], 
            'color'  : 'k',
            'weight' : 'normal',
            'size'   : 8,
            }

    fonts = {'family': ['Arial', 'Times New Roman', 'serif' ], 
            'color'  : 'k',
            'weight' : 'normal',
            'size'   : 7.5,
            }
    
    F_obs, clusters, cluster_p_values, _  = F_stats
    ave_diff = pow_list[0].mean(0) - pow_list[1].mean(0)
    times = 1e3 * epo_tfr.times

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        # freqs*times*channels
        freq_inds, time_inds, space_inds = clusters[clu_idx]

        ave_diff_mean_ch = ave_diff[..., space_inds].mean(axis=-1)

        highlight = np.nan * np.ones_like(ave_diff_mean_ch)
        highlight[freq_inds, time_inds] = ave_diff_mean_ch[freq_inds, time_inds]

        contour = np.nan * np.ones_like(highlight)
        for i in range( highlight.shape[0]):
            for j in range(highlight.shape[1]):
                if pd.isna(highlight[i][j]):
                    contour[i][j] = 0
                else:
                    contour[i][j] = 1

        # get topography for F stat
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
        freq_inds = np.unique(freq_inds)

        f_map = ave_diff[freq_inds].mean(axis=0)
        f_map = f_map[time_inds].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = epo_tfr.times[time_inds]

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=figsize)

        # create spatial mask
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epo_tfr.info, tmin=0, nave=None)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap=cmap,
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=markersize),
            scalings = dict(eeg=1),
            cbar_fmt='%3.2f',
            units=""
        )

        image = ax_topo.images[0]
        ax_topo.set_xlabel(
            cbunit+"\n"+str((int(sig_times[0]*100)*10))+"-"+str((int(sig_times[-1]*100)*10))+" ms", fontdict=font
        ) 
        
        ax_topo.set_title("")

        divider = make_axes_locatable(ax_topo)

        ax_spec = divider.append_axes("right", size="290%", pad=0.45)
        
        c = ax_spec.imshow(ave_diff_mean_ch, cmap=cmap,  #plt.cm.gray, jet, autumn plt.cm.RdBu_r
                extent=[times[0], times[-1], 0, 40], # freqs[0], freqs[-1]],
                aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

        ax_spec.contour(contour, 1,  levels=[0,1], colors="black", linewidths=1,
                    extent=[times[0], times[-1], 0, 40], # freqs[0], freqs[-1] ],
                    origin = 'lower', )#aspect = 'auto', 
        
        ax_spec.spines['top'].set_linewidth(0.5)
        ax_spec.spines['right'].set_linewidth(0.5)
        ax_spec.spines['bottom'].set_linewidth(0.5)
        ax_spec.spines['left'].set_linewidth(0.5)
        ax_spec.tick_params(which='both', bottom=True, top=False, left=True, right=False,
                labelbottom=True, labelleft=True, direction='out',width=0.5)
        
        ax_spec.xaxis.set_minor_locator(MultipleLocator(20))

        Hz = [4,8,13,30]
        ytick = [0,12,22,40]
        plt.yticks(ytick,Hz)
        ax_spec.set_yticks(ytick,Hz,fontdict=font)
        
        ax_spec.set_xticks([0,100,200,300,400,500,600,700,800,900,1000], ['0.0','','0.2','','0.4','','0.6','','0.8','','1.0'], fontdict=font);
        ax_spec.set_xlabel("Time (s)", fontdict=font)
        ax_spec.set_ylabel("Frequency (Hz)", fontdict=fonts)
        ax_spec.set_title(title, fontdict=font)

        # add another colorbar
        ax_colorbar2 = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(c, cax=ax_colorbar2)
        cbar.outline.set_linewidth(0.5)
    
        tick_locator = ticker.MaxNLocator(nbins=3) 
        cbar.locator = tick_locator
        
        if len(cbar_ticks) >0 :
            ax_colorbar2.set_yticks(cbar_ticks,cbar_ticks)
        ax_colorbar2.set_ylabel(cbunit, fontdict=font)      
        
        cbar.ax.tick_params(labelsize=8, width=0.5)
        cbar_label = cbar.ax.get_xticklabels() 
        [lab.set_fontname('Arial') for lab in cbar_label]

        plt.show()


def draw_ctd_res(res, tmin=0.0, tmax=1.0, vmin=0.0, vmax=1.0, cmap='jet', 
                 contour=-1, contour_color='k', smooth_sigma=0.5,
                 chance=0.5,
                 major_tick=0.1, minor_tick=0.05, vlines=[], hlines=[], 
                 title='', cbar_ticks=[], y_ticks=[]):
    
    font = {'family' : ['Arial', 'Times New Roman', 'serif' ], 
            'color'  : 'k',
            'weight' : 'normal',
            'size'   : 9,
            }

    fonts = {'family': ['Arial', 'Times New Roman', 'serif' ], 
            'color'  : 'k',
            'weight' : 'normal',
            'size'   : 8,
            }
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    im = ax.imshow(
        res,
        interpolation="None",
        origin="lower",
        cmap=cmap,
        extent=[tmin, tmax, tmin, tmax],
        vmin=vmin,
        vmax=vmax,
    )
    if isinstance(contour, float):
        data = gaussian_filter(res, smooth_sigma)
        ax.contour(data, [0.0, contour, 1.0], colors=contour_color, linewidths=0.75, origin='lower', extent=[tmin, tmax, tmin, tmax])
    else:
        contour_s = gaussian_filter(contour, smooth_sigma)
        ax.contour(contour_s==True, colors=contour_color, linewidths=0.75, origin='lower', extent=[tmin, tmax, tmin, tmax])

    ax.xaxis.set_major_locator(MultipleLocator(major_tick))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
    ax.yaxis.set_major_locator(MultipleLocator(major_tick))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick))

    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    ax.set_xlabel("Testing Time (s)", fontdict=font)
    ax.set_ylabel("Training Time (s)", fontdict=font)
    ax.set_title(title, fontdict=font)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("Accuracy", fontdict=font)

    if len(cbar_ticks) >0 :
        cbar.ax.set_yticks(cbar_ticks,cbar_ticks)
    
    cbar.ax.tick_params(labelsize=8)
    cbar_label = cbar.ax.get_xticklabels() 
    [lab.set_fontname('Arial') for lab in cbar_label]

    ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], ['0.0','','0.2','','0.4','','0.6','','0.8',], fontdict=fonts);
    ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], ['0.0','','0.2','','0.4','','0.6','','0.8',], fontdict=fonts);

    if len(vlines)>0:
        for l in vlines:
            ax.axvline(l, color="gray", linestyle="-", linewidth=0.5, alpha=0.8) 
    if len(hlines)>0:
        for l in hlines:
            ax.axhline(l, color="gray", linestyle="-", linewidth=0.5, alpha=0.8)

    fig, ax = plt.subplots(figsize=(1.7, 1.2))
    times = np.linspace(tmin, tmax, res.shape[0])

    res_line = savgol_filter(np.diag(res), 10, 3) 
    ax.plot(times, res_line, color='k', label="Accuracy", linewidth=0.75)
    ax.set_xlim((0, 0.8))
    
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(major_tick))

    ax.spines['top'].set_linewidth(0.0)
    ax.spines['right'].set_linewidth(0.0)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(which='both', bottom=True, top=False, left=True, right=False,
            labelbottom=True, labelleft=True, direction='out',width=0.5)
    
    ax.axhline(chance, color="k", linestyle="--", label="Chance", linewidth=0.75)
    ax.set_xlabel("Time (s)", fontdict=font)
    ax.set_ylabel("Accuracy", fontdict=font)
    ax.set_title(title, fontdict=font)

    ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], ['0.0','','0.2','','0.4','','0.6','','0.8',], fontdict=font);

    if len(y_ticks) >0 :
        ax.set_yticks(y_ticks, y_ticks, fontdict=font);
    y_label = ax.get_yticklabels() 
    [y_label_temp.set_fontname('Arial') for y_label_temp in y_label];
    [y_label_temp.set_fontsize(9) for y_label_temp in y_label];
    