import numpy as np

from astropy.io import fits


def lris_read_amp(inp, ext, redchip=False, applygain=True):
    """
    Modified from pypeit.spectrographs.keck_lris.lris_read_amp -- Jon Brown, Josh Bloom
    cf. https://github.com/KerryPaterson/Imaging_pipelines
    Read one amplifier of an LRIS multi-extension FITS image
    Parameters
    ----------
    inp: tuple
      (str,int) filename, extension
      (hdu,int) FITS hdu, extension
    Returns
    -------
    data
    predata
    postdata
    x1
    y1
    ;------------------------------------------------------------------------
    function lris_read_amp, filename, ext, $
      linebias=linebias, nobias=nobias, $
      predata=predata, postdata=postdata, header=header, $
      x1=x1, x2=x2, y1=y1, y2=y2, GAINDATA=gaindata
    ;------------------------------------------------------------------------
    ; Read one amp from LRIS mHDU image
    ;------------------------------------------------------------------------
    """
    # Parse input
    if isinstance(inp, str):
        hdu = fits.open(inp)
    else:
        hdu = inp

    # Get the pre and post pix values
    # for LRIS red POSTLINE = 20, POSTPIX = 80, PRELINE = 0, PRECOL = 12
    head0 = hdu[0].header
    precol = head0["precol"]
    postpix = head0["postpix"]

    # Deal with binning
    binning = head0["BINNING"]
    xbin, ybin = [int(ibin) for ibin in binning.split(",")]
    precol = precol // xbin
    postpix = postpix // xbin

    # get entire extension...
    temp = hdu[ext].data.transpose()  # Silly Python nrow,ncol formatting
    tsize = temp.shape
    nxt = tsize[0]

    # parse the DETSEC keyword to determine the size of the array.
    header = hdu[ext].header
    detsec = header["DETSEC"]
    x1, x2, y1, y2 = np.array(load_sections(detsec, fmt_iraf=False)).flatten()

    # parse the DATASEC keyword to determine the size of the science region (unbinned)
    datasec = header["DATASEC"]
    xdata1, xdata2, ydata1, ydata2 = np.array(
        load_sections(datasec, fmt_iraf=False)
    ).flatten()

    # grab the components...
    predata = temp[0:precol, :]
    # datasec appears to have the x value for the keywords that are zero
    # based. This is only true in the image header extensions
    # not true in the main header.  They also appear inconsistent between
    # LRISr and LRISb!
    # data     = temp[xdata1-1:xdata2-1,*]
    # data = temp[xdata1:xdata2+1, :]

    # JB: LRIS-R is windowed differently, so the default pypeit checks fail
    # xshape is calculated from datasec.
    # For blue, its 1024,
    # For red, the chip dimensions are different AND the observations are windowed
    # In windowed mode each amplifier has differently sized data sections
    if not redchip:
        xshape = 1024 // xbin  # blue
    else:
        xshape = xdata2 - xdata1 + 1 // xbin  # red

    # do some sanity checks
    if (xdata1 - 1) != precol:
        # msgs.error("Something wrong in LRIS datasec or precol")
        errStr = "Something wrong in LRIS datasec or precol"
        print(errStr)

    if (xshape + precol + postpix) != temp.shape[0]:
        # msgs.error("Wrong size for in LRIS detector somewhere.  Funny binning?")
        errStr = "Wrong size for in LRIS detector somewhere.  Funny binning?"
        print(errStr)

    data = temp[precol : precol + xshape, :]
    postdata = temp[nxt - postpix : nxt, :]

    # flip in X as needed...
    if x1 > x2:
        xt = x2
        x2 = x1
        x1 = xt
        data = np.flipud(data)  # reverse(temporary(data),1)

    # flip in Y as needed...
    if y1 > y2:
        yt = y2
        y2 = y1
        y1 = yt
        data = np.fliplr(data)
        predata = np.fliplr(predata)
        postdata = np.fliplr(postdata)

    # dummy gain data since we're keeping as uint16
    gaindata = 0.0 * data + 1.0

    return data, gaindata, predata, postdata, x1, y1


def load_sections(string, fmt_iraf=True):
    """
    Modified from pypit.core.parse.load_sections -- Jon Brown,  Josh Bloom
    cf. https://github.com/KerryPaterson/Imaging_pipelines
    From the input string, return the coordinate sections
    Parameters
    ----------
    string : str
      character string of the form [x1:x2,y1:y2]
      x1 = left pixel
      x2 = right pixel
      y1 = bottom pixel
      y2 = top pixel
    fmt_iraf : bool
      Is the variable string in IRAF format (True) or
      python format (False)
    Returns
    -------
    sections : list (or None)
      the detector sections
    """
    xyrng = string.strip("[]()").split(",")
    if xyrng[0] == ":":
        xyarrx = [0, 0]
    else:
        xyarrx = xyrng[0].split(":")
        # If a lower/upper limit on the array slicing is not given (e.g. [:100] has no lower index specified),
        # set the lower/upper limit to be the first/last index.
        if len(xyarrx[0]) == 0:
            xyarrx[0] = 0
        if len(xyarrx[1]) == 0:
            xyarrx[1] = -1
    if xyrng[1] == ":":
        xyarry = [0, 0]
    else:
        xyarry = xyrng[1].split(":")
        # If a lower/upper limit on the array slicing is not given (e.g. [5:] has no upper index specified),
        # set the lower/upper limit to be the first/last index.
        if len(xyarry[0]) == 0:
            xyarry[0] = 0
        if len(xyarry[1]) == 0:
            xyarry[1] = -1
    if fmt_iraf:
        xmin = max(0, int(xyarry[0]) - 1)
        xmax = int(xyarry[1])
        ymin = max(0, int(xyarrx[0]) - 1)
        ymax = int(xyarrx[1])
    else:
        xmin = max(0, int(xyarrx[0]))
        xmax = int(xyarrx[1])
        ymin = max(0, int(xyarry[0]))
        ymax = int(xyarry[1])
    return [[xmin, xmax], [ymin, ymax]]


def sec2slice(
    subarray, one_indexed=False, include_end=False, require_dim=None, transpose=False
):
    """
    Modified from pypit.core.parse.sec2slice -- Jon Brown
    Convert a string representation of an array subsection (slice) into
    a list of slice objects.
    Args:
        subarray (str):
            The string to convert.  Should have the form of normal slice
            operation, 'start:stop:step'.  The parser ignores whether or
            not the string has the brackets '[]', but the string must
            contain the appropriate ':' and ',' characters.
        one_indexed (:obj:`bool`, optional):
            The string should be interpreted as 1-indexed.  Default
            is to assume python indexing.
        include_end (:obj:`bool`, optional):
            **If** the end is defined, adjust the slice such that
            the last element is included.  Default is to exclude the
            last element as with normal python slicing.
        require_dim (:obj:`int`, optional):
            Test if the string indicates the slice along the proper
            number of dimensions.
        transpose (:obj:`bool`, optional):
            Transpose the order of the returned slices.  The
            following are equivalent::
                tslices = parse_sec2slice('[:10,10:]')[::-1]
                tslices = parse_sec2slice('[:10,10:]', transpose=True)
    Returns:
        tuple: A tuple of slice objects, one per dimension of the
        prospective array.
    Raises:
        TypeError:
            Raised if the input `subarray` is not a string.
        ValueError:
            Raised if the string does not match the required
            dimensionality or if the string does not look like a
            slice.
    """
    # Check it's a string
    if not isinstance(subarray, (str, bytes)):
        raise TypeError("Can only parse string-based subarray sections.")
    # Remove brackets if they're included
    sections = subarray.strip("[]").split(",")
    # Check the dimensionality
    ndim = len(sections)
    if require_dim is not None and ndim != require_dim:
        raise ValueError(
            "Number of slices ({0}) in {1} does not match ".format(ndim, subarray)
            + "required dimensions ({0}).".format(require_dim)
        )
    # Convert the slice of each dimension from a string to a slice
    # object
    slices = []
    for s in sections:
        # Must be able to find the colon
        if ":" not in s:
            raise ValueError("Unrecognized slice string: {0}".format(s))
        # Initial conversion
        _s = [None if x == "" else int(x) for x in s.split(":")]
        if len(_s) > 3:
            raise ValueError(
                "String as too many sections.  Must have format 'start:stop:step'."
            )
        if len(_s) < 3:
            # Include step
            _s += [None]
        if one_indexed:
            # Decrement to convert from 1- to 0-indexing
            _s = [None if x is None else x - 1 for x in _s]
        if include_end and _s[1] is not None:
            # Increment to include last
            _s[1] += 1
        # Append the new slice
        slices += [slice(*_s)]
    return tuple(slices[::-1] if transpose else slices)

def read_lris(hdul, det=None, TRIM=False):
    """
    Modified from pypeit.spectrographs.keck_lris.read_lris -- Jon Brown, Josh Bloom
    cf. https://github.com/KerryPaterson/Imaging_pipelines
    Read a raw LRIS data frame (one or more detectors)
    Packed in a multi-extension HDU
    Based on readmhdufits.pro
    Parameters
    ----------
    raw_file : str
      Filename
    det : int, optional
      Detector number; Default = both
    TRIM : bool, optional
      Trim the image?
    Returns
    -------
    array : ndarray
      Combined image
    header : FITS header
    sections : list
      List of datasec, oscansec, ampsec sections
    """
    head0 = hdul[0].header

    # Get post, pre-pix values
    precol = head0["PRECOL"]
    postpix = head0["POSTPIX"]
    preline = head0["PRELINE"]
    postline = head0["POSTLINE"]

    # get the detector
    # this just checks if its the blue one and assumes red if not
    # note the red fits headers don't even have this keyword???
    if head0["INSTRUME"] == "LRISBLUE":
        redchip = False
    else:
        redchip = True

    # Setup for datasec, oscansec
    dsec = []
    osec = []
    nxdata_sum = 0

    # get the x and y binning factors...
    binning = head0["BINNING"]
    xbin, ybin = [int(ibin) for ibin in binning.split(",")]

    # First read over the header info to determine the size of the output array...
    n_ext = len(hdul) - 1  # Number of extensions (usually 4)
    xcol = []
    xmax = 0
    ymax = 0
    xmin = 10000
    ymin = 10000
    for i in np.arange(1, n_ext + 1):
        theader = hdul[i].header
        detsec = theader["DETSEC"]
        if detsec != "0":
            # parse the DETSEC keyword to determine the size of the array.
            x1, x2, y1, y2 = np.array(load_sections(detsec, fmt_iraf=False)).flatten()

            # find the range of detector space occupied by the data
            # [xmin:xmax,ymin:ymax]
            xt = max(x2, x1)
            xmax = max(xt, xmax)
            yt = max(y2, y1)
            ymax = max(yt, ymax)

            # find the min size of the array
            xt = min(x1, x2)
            xmin = min(xmin, xt)
            yt = min(y1, y2)
            ymin = min(ymin, yt)
            # Save
            xcol.append(xt)

    # determine the output array size...
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1

    # change size for binning...
    nx = nx // xbin
    ny = ny // ybin

    # Update PRECOL and POSTPIX
    precol = precol // xbin
    postpix = postpix // xbin

    # Deal with detectors
    if det in [1, 2]:
        nx = nx // 2
        n_ext = n_ext // 2
        det_idx = np.arange(n_ext, dtype=np.int) + (det - 1) * n_ext
    elif det is None:
        det_idx = np.arange(n_ext).astype(int)
    else:
        raise ValueError("Bad value for det")

    # change size for pre/postscan...
    if not TRIM:
        nx += n_ext * (precol + postpix)
        ny += preline + postline

    # allocate output array...
    array = np.zeros((nx, ny), dtype="uint16")
    gain_array = np.zeros((nx, ny), dtype="uint16")
    order = np.argsort(np.array(xcol))

    # insert extensions into master image...
    for kk, i in enumerate(order[det_idx]):

        # grab complete extension...
        data, gaindata, predata, postdata, x1, y1 = lris_read_amp(
            hdul, i + 1, redchip=redchip
        )

        # insert components into output array...
        if not TRIM:
            # insert predata...
            buf = predata.shape
            nxpre = buf[0]
            xs = kk * precol
            xe = xs + nxpre

            array[xs:xe, :] = predata
            gain_array[xs:xe, :] = predata

            # insert data...
            buf = data.shape
            nxdata = buf[0]
            nydata = buf[1]

            # JB: have to track the number of xpixels
            xs = n_ext * precol + nxdata_sum
            xe = xs + nxdata

            # now log how many pixels that was
            nxdata_sum += nxdata

            # Data section
            # section = '[{:d}:{:d},{:d}:{:d}]'.format(preline,nydata-postline, xs, xe)  # Eliminate lines
            section = "[{:d}:{:d},{:d}:{:d}]".format(
                preline, nydata, xs, xe
            )  # DONT eliminate lines

            dsec.append(section)
            array[xs:xe, :] = data  # Include postlines
            gain_array[xs:xe, :] = gaindata  # Include postlines

            # ; insert postdata...
            buf = postdata.shape
            nxpost = buf[0]
            xs = nx - n_ext * postpix + kk * postpix
            xe = xs + nxpost
            section = "[:,{:d}:{:d}]".format(xs, xe)
            osec.append(section)

            array[xs:xe, :] = postdata
            gain_array[xs:xe, :] = postdata

        else:
            buf = data.shape
            nxdata = buf[0]
            nydata = buf[1]

            xs = (x1 - xmin) // xbin
            xe = xs + nxdata
            ys = (y1 - ymin) // ybin
            ye = ys + nydata - postline

            yin1 = preline
            yin2 = nydata - postline

            array[xs:xe, ys:ye] = data[:, yin1:yin2]
            gain_array[xs:xe, ys:ye] = gaindata[:, yin1:yin2]

    # make sure BZERO is a valid integer for IRAF
    obzero = head0["BZERO"]
    head0["O_BZERO"] = obzero
    head0["BZERO"] = 32768 - obzero

    # Return, transposing array back to goofy Python indexing
    return array.T, head0
