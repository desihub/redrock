========================
Priors format in Redrock
========================

Implementation
--------------

The implementation was done in (PR `#152`_) and updated in (PR `#194`_)

.. _`#152`: https://github.com/desihub/redrock/pull/152
.. _`#194`: https://github.com/desihub/redrock/pull/194


Prior form available
--------------------

* gaussian    --> the amplitude is set to one. I (Edmond) noticed it is not always sufficient --> in this case use tophat prior
* lorentzien  --> same remark than above
* tophat      --> useful if you really trust the region where you want to search. Use for the clustering QSO catalog in DESI


Use prior in RR
---------------

* Build a prior file as explained above. The prior_file as to contain a prior for every targetid on which redrock will be run
* Use the flag : --priors filename_priors during the execution of rrdesi

Prior file
----------

* I (Edmond) give here a minimal function to write in a correct way the prior file:

.. code-block:: python
    import numpy as np
    import fitsio

    def write_prior_for_RR(targetid, z_prior, filename_priors):
        """
            Minimal fonction to write prior file for redrock.

            targetid : must be the array of targetid list given to redrock in the rrdesi command
            z_prior  : array of size targetid.size containing the prior value of the redshift for the considered targetid. For instant value from QuasarNet.
            filename_priors : name of the prior file which will be given to the rrdesi command

        """

        # need to be the same for every target
        # only function[0] will be read in the prior class !
        function = np.array(['tophat'] * z_prior.size)

        # can be different for every target (I set it constant here)
        sigma = 0.1*np.ones(z_prior.size)

        # save
        out = fitsio.FITS(filename_priors, 'rw', clobber=True)
        data, names, extname = [targetid, function, z_prior, sigma], ['TARGETID', 'FUNCTION', 'Z', 'SIGMA'], 'PRIORS'
        out.write(data, names=names, extname=extname)
        out.close()

        print(f'     Write prior file for RR with {z_prior.size} objetcs: {filename_priors}')
