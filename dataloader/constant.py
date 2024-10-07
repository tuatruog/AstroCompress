class DatasetConstants:
    # 2D dataset name
    JWST = 'jwst'
    KECK = 'keck'
    SDSS = 'sdss'
    LCO = 'lco'
    HST = 'hst'
    HST_5 = 'hst-5'  # hst dataset with only 5 test image
    SDSS_10 = 'sdss-10'  # sdss dataset with only 10 test image

    # 3D Residual dataset
    JWST_RES = 'jwst-res'
    JWST_RES_1 = 'jwst-res1'
    SDSS_RES = 'sdss-res'

    # 3D dataset
    SDSS_3D = 'sdss-3d'  # image with the 3rd dimension being time dimension
    SDSS_3T = 'sdss-3t'

    # 4D Residual dataset
    SDSS_4D = 'sdss-4d'  # image with multiple channels
    SDSS_4D_RES = 'sdss-4d-res'  # residual multi-channel image between different timestep

    # Full fits image dataset
    JWST_FULL = 'jwst-full'
    SDSS_FULL = 'sdss-full'

    DATASETS = [JWST, KECK, SDSS, LCO, HST, HST_5, SDSS_10, JWST_RES,
                     JWST_RES_1, SDSS_RES, JWST_FULL, SDSS_FULL, SDSS_3D, SDSS_3T,
                     SDSS_4D, SDSS_4D_RES]

    DATASET_NUM_CHANNELS = {
        JWST:1, KECK:1, SDSS:1, LCO:1, HST:1, JWST_RES:1, SDSS_RES:1,
        JWST_FULL:'variable', SDSS_FULL:'variable',     # These vary across different fits files.
        SDSS_3D:3, SDSS_3T:3,
        SDSS_4D:3, SDSS_4D_RES:3
    }

    # Hugging face dataset repo
    ASTRO_COMPRESS_REPO = 'AstroCompress/'

    SBI_16_2D = 'SBI-16-2D'
    SBI_16_3D = 'SBI-16-3D'
    GBI_16_2D = 'GBI-16-2D'
    GBI_16_4D = 'GBI-16-4D'
    GBI_16_2D_LEGACY = 'GBI-16-2D-Legacy'

    ASTRO_COMPRESS_SBI_16_2D = ASTRO_COMPRESS_REPO + SBI_16_2D
    ASTRO_COMPRESS_SBI_16_3D = ASTRO_COMPRESS_REPO + SBI_16_3D
    ASTRO_COMPRESS_GBI_16_2D = ASTRO_COMPRESS_REPO + GBI_16_2D
    ASTRO_COMPRESS_GBI_16_4D = ASTRO_COMPRESS_REPO + GBI_16_4D
    ASTRO_COMPRESS_GBI_16_2D_LEGACY = ASTRO_COMPRESS_REPO + GBI_16_2D_LEGACY
