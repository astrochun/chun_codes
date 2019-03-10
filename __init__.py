def systime():
  import time
  return time.strftime("%d_%b_%Y_%H:%M:%S", time.localtime())
#enddef

def match_nosort(a, b, uniq=False):
# Modified on 06/04/2016 to include uniq.
  import numpy as np
  subb = np.repeat(-1, len(a))

  for ii in range(len(a)):
    mark = np.where(b == a[ii])
    if len(mark[0]) == 1: subb[ii] = mark[0]
    if len(mark[0]) >= 2:
      if uniq == True: subb[ii] = mark[0][0]
  #endfor

  #print subb
  suba = (np.where(subb != -1))[0]
  subb = subb[suba]

  return suba, subb
#enddef

### Not fully tested
def match_nosort_str(a, b):
  import numpy as np

  subb = np.repeat(-1, len(a))

  suba = np.array([i for i, v in enumerate(a) if v in set(b)])
  
  for ii in range(len(a)):
    mark = [xx for xx in range(len(b)) if a[ii] in b[xx]]
    if len(mark) == 1: subb[ii] = mark[0]
  #endfor                      
  #
  #print subb                  
  suba = (np.where(subb != -1))[0]
  subb = subb[suba]
  #
  return suba, subb
#enddef               

### + on 04/03/2016
def intersect(a, b):
  import numpy as np
  return np.array(list(set(a) & set(b)))

def chun_crossmatch(x1, y1, x2, y2, dcr, **kwargs):
  # Mod on 23/04/2016 to fix ind1,ind2 if no return is made
  import numpy as np

  silent = 0
  if kwargs and kwargs.has_key('silent'): silent = 1
  if not silent: print('### Begin chun_crossmatch '+systime())

  verbose = 0
  if kwargs and kwargs.has_key('verbose'): verbose = 1

  sph = 0
  if kwargs and kwargs.has_key('sph'): sph = 1

  len0 = len(x1)
# print len0

  cnt = long(0)
  for ii in range(len0):
    if verbose and len0 >= 100000:
      if ii%1000 == 0: print('ii = '+strn(ii, f='(I)'), systime())

    if sph == 1:
      xdiff = (x1[ii] - x2)*3600.0*np.cos(y1[ii]*np.pi/180.0)
      ydiff = (y1[ii] - y2)*3600.0
    else:
      xdiff = x1[ii] - x2
      ydiff = y1[ii] - y2

    distance = np.sqrt(xdiff**2 + ydiff**2)

    inreg = (np.where(distance <= dcr))[0]
    #print ii, inreg
    #inreg = [xx for xx in range(len(distance)) if (distance[xx] <= dcr)]

    if len(inreg) > 0:
      min = [xx for xx in range(len(distance)) if 
             (distance[xx] == np.min(distance[inreg]))]
      if cnt == 0:
        save   = min
        zsave  = [ii]
        dx = [xdiff[min]]
        dy = [ydiff[min]]
      else:
        zsave.append(ii)

        if len(min) == 1:
          save.append(min[0])
          dx.append(xdiff[min])
          dy.append(ydiff[min])
        else:
          for jj in range(len(min)):
            save.append(min[jj])
            dx.append(xdiff[min[jj]])
            dy.append(ydiff[min[jj]])
          #endfor
        #endelse
      cnt = cnt+1
    #endif
  #endfor

  if (cnt == 0):
      ind1 = np.array([-1])
      ind2 = np.array([-1])
  else:
      ind1 = np.array(zsave)
      ind2 = np.array(save)

  if not silent: print('### End chun_crossmatch '+systime())
  return ind1, ind2
#enddef

def ds9_reg(XX, YY, ds9_file, color='green', aper=[2.0], image=False, wcs=False, file0=''):
  import numpy as np

  if color != 'green':
    color_str = ' # color = '+color
  else: color_str = '' 
 
  if image == True:
     coord = 'physical'
     suff0 = ')' + color_str

  if wcs == True:
     coord = 'fk5'
     suff0 = '")'+color_str

  if len(aper) == 1:
    aper0 = np.repeat(aper[0], len(XX))
  else: aper0 = aper

  str0 = ['# Region file format: DS9 version 4.0', '# Filename: '+file0,
          ' global color=green font="helvetica 10 normal" select=1 '+
          'highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source', coord]

  #print str0
  print '### Writing : ', ds9_file
  f = file(ds9_file, 'w')
  for jj in range(len(str0)): f.write(str0[jj]+'\n')
  for ii in range(len(XX)):
    txt = 'circle(%f,%f,%f%s\n' % (XX[ii], YY[ii], aper0[ii], suff0)
    f.write(txt)

  f.close()
#enddef

def random_pdf(x, dx, seed_i=False, n_iter=1000, silent=True):
  ## Added on 24/06/2016
  ## Mod on 29/06/2016 to reverse shape

  import numpy as np

  len0 = len(x)
  if silent == False: print len0

  # Mod on 29/06/2016
  x_pdf  = np.zeros((len0, n_iter), dtype=np.float64)
  #x_pdf = np.zeros((n_iter, len0), dtype=np.float64)

  if seed_i != False:
    seed0 = seed_i + np.arange(len0)
  
  for ii in range(len0):
    if seed_i == False:
      temp = np.random.normal(0.0, 1.0, size=n_iter)
    else:
      np.random.seed(seed0[ii])
      temp = np.random.normal(0.0, 1.0, size=n_iter)

    rand_ans  = x[ii] + dx[ii]*temp
    x_pdf[ii] = rand_ans
  #endfor

  return x_pdf
#enddef

def compute_onesig_pdf(arr0, x_val, usepeak=False, silent=True, verbose=False):
  ### + on 28/06/2016
  ### Mod on 29/06/2016 to handle change in shape

  import numpy as np

  if silent == False: print '### Begin compute_onesig_pdf | '+systime()

  len0 = arr0.shape[0] # arr0.shape[1] # Mod on 29/06/2016

  err   = np.zeros((len0,2)) # np.zeros((2,len0)) # Mod on 29/06/2016
  xpeak = np.zeros(len0)

  conf = 0.68269 # 1-sigma

  for ii in range(len0):
    test = arr0[ii] # arr0[:,ii] # Mod on 29/06/2016
    good = np.where(np.isfinite(test) == True)[0]
    if len(good) > 0:
      v_low  = np.percentile(test[good],15.8655)
      v_high = np.percentile(test[good],84.1345)

      xpeak[ii] = np.percentile(test[good],50.0)
      if usepeak == False:
        t_ref = x_val[ii]
      else:
        t_ref = xpeak[ii]

      err[ii,0]  = t_ref - v_low
      err[ii,1]  = v_high - t_ref
      #err[0,ii] = t_ref - v_low
      #err[1,ii] = v_high - t_ref
        
      #sig0 = np.std(test[good])
      #if sig0 != 0.0:
      #  datamin = np.min(test[good])
      #  datamax = np.max(test[good])
      #  numbins = np.ceil((datamax-datamin)/(sig0/5.0))
      #  mybins  = np.linspace(datamin, datamax, numbins)
      #  htemp, jnk = np.histogram(temp[good], mybins)
      #  
      #  peak      = np.max(htemp)
      #  xpeak[ii] = mybins[np.where(htemp == peak)[0]]
    #endif
  #endfor
  if silent == False: print '### End compute_onesig_pdf | '+systime()
  return err, xpeak
#enddef

#  if len(save) N_elements(save) gt N_elements(zsave) then begin
#     print, '### Error: (x2,y2) may have duplicate position entries. '+systime()
#     print, '### Exiting!!!'
#     return
#  endif
#enddef

def plot_data_err_hist(x, dx, xlabel, out_pdf, c0='b', m0='o', a0=0.5, s0=25,
                       x_bins=50, y_bins=50, xlim=None, ylim=None):
    '''

    Generate plot of variable and uncertainty on variable. This produces
    a three-panel PDF with histograms on variable at the bottom and
    uncertainty on the right side (oriented 90deg)

    Parameters
    ----------
    x : array_like
        An array or arrays of variable (N or N_var x N)

    dx : array_like
        An array or arrays of uncertainty for x (N or N_var x N)

    out_pdf : file path, file object, or file like object
        File to write to.  If opened, must be opened for append (ab+).

    c0 : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs
        (see below). Note that `c` should not be a single numeric RGB or
        RGBA sequence because that is indistinguishable from an array of
        values to be colormapped.  `c` can be a 2-D array in which the
        rows are RGB or RGBA, however.

    m0 : `~matplotlib.markers.MarkerStyle`, optional, default: 'o'
        See `~matplotlib.markers` for more information on the different
        styles of markers scatter supports.

    a0 : scalar, optional, default: None
        The alpha blending value, between 0 (transparent) and 1 (opaque)

    x_bins : integer or array_like for x, optional, default: 50
        If an integer is given, `bins + 1` bin edges are returned,
        consistently with :func:`numpy.histogram` for numpy version >=
        1.3.

    y_bins : integer or array_like for dx, optional, default: 50
        If an integer is given, `bins + 1` bin edges are returned,
        consistently with :func:`numpy.histogram` for numpy version >=
        1.3.

    xlim : array_like, optional, default: None
        limits for x

    ylim : array_like, optional, default: None
        limits for dx

    Notes
    -----
        Created by Chun Ly on 29 June 2016
        Additional modification to handle multiple variables
    '''

    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    from pylab import subplots_adjust
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages # + on 29/06/2016

    if x.ndim == 1:
      x  = x.reshape((1,len(x)))
      dx = dx.reshape((1,len(dx)))
      xlabel = [xlabel]
      xlim = np.array(xlim)
      xlim = xlim.reshape((1,2))
      ylim = np.array(ylim)
      ylim = ylim.reshape((1,2))

    n_var = x.shape[0]

    pp = PdfPages(out_pdf)

    for ii in range(n_var):
      gs  = gridspec.GridSpec(3, 3)
      ax1 = plt.subplot(gs[:-1,:-1])
      ax2 = plt.subplot(gs[2,:-1])
      ax3 = plt.subplot(gs[:-1,2])

      # Panel 1

      # Get number of sources within region
      if xlim != None and ylim != None:
        in_field = [(xlim[ii,0] <= a <= xlim[ii,1] and
                     ylim[ii,0] <= b <= ylim[ii,1]) for a,b in
                    zip(x[ii],dx[ii])]

      ax1_label = 'N='+str(sum(in_field))
      ax1.scatter(x[ii], dx[ii], c=c0, marker=m0, s=s0, alpha=a0,
                  edgecolor='none', label=ax1_label)
      ax1.xaxis.set_ticklabels([])

      if xlim != None: ax1.set_xlim(xlim[ii])
      if ylim != None: ax1.set_ylim(ylim[ii])

      ylabel = r'$\sigma$('+xlabel[ii]+')'
      ax1.set_ylabel(ylabel)

      ax1.legend(loc='lower right', fontsize='12', scatterpoints=3,
                 frameon=False)
    
      # Lower histogram
      ax2.hist(x[ii], bins=x_bins, fc=c0, alpha=a0, histtype='stepfilled',
               edgecolor='None')
      ax2.hist(x[ii], bins=x_bins, fc='None', histtype='stepfilled',
               edgecolor=c0, lw=1.5)

      avg0 = '%.3f ' % np.average(x[ii])
      med0 = '%.3f ' % np.median(x[ii])
      sig0 = '%.3f ' % np.std(x[ii])
      txt0 = 'Average : '+avg0+'\n'+'Median : '+med0+'\n'+r'$\sigma$ : '+sig0
      ax2.annotate(txt0, (0.97,0.97), xycoords='axes fraction', ha='right',
                   va='top')

      ax2.set_xlabel(xlabel[ii])
      ax2.set_ylabel('N')
    
      if xlim != None:
        ax2.set_xlim(xlim[ii])
      else: ax2.set_xlim(ax1.get_xlim())
      
      # Right histogram
      ax3.hist(dx[ii], bins=y_bins, orientation='horizontal', fc=c0, alpha=a0,
               histtype='stepfilled', edgecolor='None')
      ax3.hist(dx[ii], bins=y_bins, orientation='horizontal', fc='None',
               histtype='stepfilled', edgecolor=c0, lw=1.5)
      ax3.yaxis.set_ticklabels([])
      
      ax3.set_xlabel('N')
      
      avg0 = '%.3f ' % np.average(dx[ii])
      med0 = '%.3f ' % np.median(dx[ii])
      txt0 = 'Average : '+avg0+'\n'+'Median : '+med0
      ax3.annotate(txt0, (0.94,0.03), xycoords='axes fraction',
                   ha='right', va='bottom')
      
      if ylim != None:
        ax3.set_ylim(ylim[ii])
      else: ax3.set_ylim = (ax1.get_ylim())
      
      # Tick marks
      ax1.minorticks_on()
      ax2.minorticks_on()
      ax3.minorticks_on()
      
      subplots_adjust(left=0.01, bottom=0.01, top=0.99, right=0.99,
                      wspace=0.05, hspace=0.05)
      
      fig = plt.gcf()
      fig.set_size_inches(8,8)

      fig.savefig(pp, format='pdf', bbox_inches='tight')
      #fig.savefig(out_pdf, bbox_inches='tight')
      plt.close()
      fig.clear()
    #endfor

    print '### Writing : ', out_pdf
    pp.close()
#enddef

def quad_low_high_err(err, hi=None):
  import numpy as np

  if hi == None:
    return np.sqrt((err[:,0]**2 + err[:,1]**2)/2.0)
  else:
    return np.sqrt((err**2 + hi**2)/2.0)

#enddef

def plot_compare(x0, y0, out_pdf, labels, extra_label=['',''], idx=None,
                 x0_err=None, y0_err=None, xlim=None, ylim=None, c0='b',
                 m0='o', a0=0.5, s0=25, silent=False, verbose=True):
    '''
    Provide explanation for function here.

    Parameters
    ----------
    x0 : array_like
        An array or arrays of variable (N or N_var x N)

    y0 : array_like
        An array or arrays of variable (N or N_var x N)

    out_pdf : file path, file object, or file like object
        File to write to.  If opened, must be opened for append (ab+).

    labels : string array
        An array or labeling for x- and y-axes with dimension of N_var

    extra_label : string array (optional)
        Additional string to append to labels for x- (1st entry) and
        y- (2nd entry) axes

    idx : array like
        Indexing array for sources to determine determine average, median.
        This allows you to exclude NaNs or unavailable values.
        Default: All sources included

    x0_err : array_like
        An array or arrays of uncertainty for x0 (N or N_var x N)

    y0_err : array_like
        An array or arrays of uncertainty for y0 (N or N_var x N)

    xlim : array_like, optional, default: None
        limits for upper panel (x0 vs y0)

    ylim : array_like, optional, default: None
        y-limit for lower panel (difference)

    c0 : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs
        (see below). Note that `c` should not be a single numeric RGB or
        RGBA sequence because that is indistinguishable from an array of
        values to be colormapped.  `c` can be a 2-D array in which the
        rows are RGB or RGBA, however.

    m0 : `~matplotlib.markers.MarkerStyle`, optional, default: 'o'
        See `~matplotlib.markers` for more information on the different
        styles of markers scatter supports.

    a0 : scalar, optional, default: 0.5
        The alpha blending value, between 0 (transparent) and 1 (opaque)

    s0 : scalar or array_like, shape (n, ), optional, default: 25
        size in points^2.
 
    silent : boolean
          Turns off stdout messages. Default: False

    verbose : boolean
          Turns off additional stdout messages. Default: True
	  
    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 15 July 2016
    Modified by Chun Ly, 18 July 2016
     - Minor modification for limit for [in_field]
    '''
    
    if silent == False: print '### Begin plot_compare | '+systime()
    
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    from pylab import subplots_adjust
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    if x0.ndim == 1:
      x0  = x0.reshape((1,len(x0)))
      y0  = y0.reshape((1,len(y0)))
      if xlim != None:
        xlim = np.array(xlim)
        xlim = xlim.reshape((1,2))
      if ylim != None:
        ylim = np.array(ylim)
        ylim = ylim.reshape((1,2))

    n_var = x0.shape[0]

    if idx == None: idx = range(x0.shape[1])

    pp = PdfPages(out_pdf)

    for ii in range(n_var):
      gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
      ax1 = plt.subplot(gs[0])
      ax2 = plt.subplot(gs[1])

      # Top Panel #

      # Get number of sources within region
      if xlim != None and ylim != None:

        # Mod on 18/07/2016. Bug found with y range. ylim -> xlim
        in_field = [(xlim[ii,0] <= a <= xlim[ii,1] and
                     xlim[ii,0] <= b <= xlim[ii,1]) for a,b in
                    zip(x0[ii],y0[ii])]
        
        ax1_label = 'N='+str(sum(in_field))

      ax1.plot(xlim[ii], xlim[ii], 'r--')

      ax1.scatter(x0[ii], y0[ii], c=c0, marker=m0, s=s0, alpha=a0,
                  edgecolor='none', label=ax1_label)

      ax1.xaxis.set_ticklabels([])

      if xlim != None:
        ax1.set_xlim(xlim[ii])
        ax1.set_ylim(xlim[ii])

      ylabel = labels[ii] + extra_label[1] #' (re-derived)'
      ax1.set_ylabel(ylabel)

      ax1.legend(loc='lower right', fontsize='12', scatterpoints=3,
                 frameon=False)
    

      # Bottom Panel #

      ax2.plot(xlim[ii], [0,0], 'r--')

      diff0 = y0[ii] - x0[ii]
      ax2.scatter(x0[ii], diff0, c=c0, marker=m0, s=s0, alpha=a0,
                  edgecolor='none')

      avg0 = '%.3f ' % np.average(diff0[idx])
      med0 = '%.3f ' % np.median(diff0[idx])
      sig0 = '%.3f ' % np.std(diff0[idx])
      txt0 = 'Average : '+avg0+'\n'+'Median : '+med0+'\n'+r'$\sigma$ : '+sig0
      ax2.annotate(txt0, (0.97,0.97), xycoords='axes fraction', ha='right',
                   va='top')

      xlabel = labels[ii] + extra_label[0] #' (published)'
      ax2.set_xlabel(xlabel)
      ax2.set_ylabel('diff.')
    
      if xlim != None:
        ax2.set_xlim(xlim[ii])
      else: ax2.set_xlim(ax1.get_xlim())

      if ylim != None:
        ax2.set_ylim(ylim[ii])
         
      ax1.minorticks_on()
      ax2.minorticks_on()

      subplots_adjust(left=0.01, bottom=0.01, top=0.99, right=0.99,
                      wspace=0.03, hspace=0.03)
      
      fig = plt.gcf()
      fig.set_size_inches(8,8)

      fig.savefig(pp, format='pdf', bbox_inches='tight')
      plt.close()
      fig.clear()
    #endfor

    print '### Writing : ', out_pdf
    pp.close()

    if silent == False: print '### End plot_compare | '+systime()
#enddef

def rem_dup(values):
  # + on 18/08/2016
  import numpy as np
  output = []
  seen = set()
  for value in values:
    if value not in seen:
      output.append(value)
      seen.add(value)
  return np.array(output)

#enddef

def gauss2d((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    2-D Gaussian for opt.curve_fit()

    Parameters
    ----------
    (x,y) : numpy.ndarray
      x,y grid from numpy.meshgrid()

    amplitude : float
      Peak of Gaussian

    xo : float
      Gaussian center value along x

    yo : float
      Gaussian center value along y

    sigma_x : float
      Gaussian sigma along x

    sigma_y : float
      Gaussian sigma along y

    theta : float
      Orientation along major axis of Gaussian. Positive is clock-wise.

    offset : float
      Level of continuum

    Returns
    -------
    g.ravel() : numpy.ndarray
      Contiguous flattened array

    Notes
    -----
    Created by Chun Ly, 26 April 2017
     - Copied from MMTtools.mmtcam for more general use
    Modified by Chun Ly, 6 May 2017
     - Fix bug. Need to import numpy
    '''
    import numpy as np

    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()
#enddef

def exec_pdfmerge(files, pages, outfile, merge=False, silent=False, verbose=True):
    '''
    Executes pdfmerge command to grab necessary pages and merge them if desired

    Require installing pdfmerge:
    https://pypi.python.org/pypi/pdfmerge/0.0.7
      > pip install pdfmerge

    Parameters
    ----------
    files : list
      List of files (must include full path)

    pages : list
      List of strings indicating pages to extract.
      E.g.: ['4,6,15,20','3,8,44,50']

    outfile : list or str
      Either List of files to write or a single file if merge == True

    silent : boolean
      Turns off stdout messages. Default: False

    verbose : boolean
      Turns on additional stdout messages. Default: True

    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 22 January 2018
    '''

    import pdfmerge
    from astropy import log

    if merge == False:
      if len(outfile) != len(files):
        log.warn('### outfile input not complete. Missing files!')

    if silent == False: log.info('### Begin exec_pdfmerge : '+systime())

    n_files = len(files)

    writer0 = None

    for nn in range(n_files):
      writer0 = pdfmerge.add(files[nn], rules=pages[nn], writer=writer0)

      if merge == False:
        with open(outfile[nn], 'wb') as stream:
          writer0.write(stream)
        writer0 = None
    #endfor

    if merge == True:
      with open(outfile, 'wb') as stream:
        writer0.write(stream)

    if silent == False: log.info('### End exec_pdfmerge : '+systime())
#enddef
