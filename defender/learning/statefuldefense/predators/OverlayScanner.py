import pefile

from learning.statefuldefense.predators.PredatorInterface import PredatorInterface


class OverlayScanner(PredatorInterface):
    """
    Checks if overlay is too large relative to file size.
    If we obtain a corrupt header, we also return True = suspicious.
    """

    def __init__(self, min_ratio: float, verbose: bool):
        """
        min_ratio: if ratio of overlay-size/file-size > min_ratio, this class returns 'suspicious'.
        """
        self.min_ratio: float = min_ratio
        self.verbose: bool = verbose
        self.use_sec_overlay: bool = True

    # @Overwrite
    def check_file(self, bytez, lief_binary) -> bool:

        pefile_file = pefile.PE(data=bytez, fast_load=True)

        # print(lief_binary.virtual_size)
        offset_overlay = pefile_file.get_overlay_data_start_offset()
        hasoverlay = True
        if offset_overlay is None:
            offset_overlay = len(bytez)
            hasoverlay = False

        # In additon, we should check for the skipped data directory by get_overlay_data_start_offset
        try:
            secdatadir = pefile_file.get_offset_from_rva(pefile_file.OPTIONAL_HEADER.DATA_DIRECTORY[4].VirtualAddress)
            sec_overlay = secdatadir + pefile_file.OPTIONAL_HEADER.DATA_DIRECTORY[4].Size
            if sec_overlay < offset_overlay:
                sec_overlay = offset_overlay
        except Exception as e:
            corruptheader = "data at RVA can't be fetched. Corrupt header?" in str(e)
            if self.verbose is True:
                print("Error in OPTIONAL_HEADER DATA DIR retrieval: " + str(e) + ";CorruptHeader:" + str(corruptheader))
            if corruptheader is True:
                return True

            sec_overlay = len(bytez)
            # TODO actually also a feature, if header is somehow broken?

        if self.verbose is True:
            print("HasOverlay:{0:<5} Offset: {1:<12} LenBytez: {2:<12} VZ:{3:<12}".format(hasoverlay, offset_overlay,
                  len(bytez), lief_binary.virtual_size))

        if self.use_sec_overlay is True:
            percentage_overlay_filesize_withsecdir = (len(bytez) - sec_overlay) / len(bytez)
            return percentage_overlay_filesize_withsecdir >= self.min_ratio
        else:
            percentage_overlay_filesize = (len(bytez) - offset_overlay) / len(bytez)
            return percentage_overlay_filesize >= self.min_ratio
