import awkward as ak
import numpy as np
import numba


@numba.njit
def _maxHistoryPDGID(id_array, mom_array, counts):
    maxPDGID_array = np.ones(len(id_array), np.int32) * -9

    # offset is the starting index for this event
    offset = 0
    # i is the event number
    for i in range(len(counts)):
        # j is the gen particle within event i
        for j in range(counts[i]):
            maxPDGID_array[offset + j] = id_array[offset + j]
            idx = mom_array[offset + j]
            while idx != -1:
                maxPDGID_array[offset + j] = max(
                    id_array[offset + idx], maxPDGID_array[offset + j]
                )
                idx = mom_array[offset + idx]
        offset += counts[i]

    return maxPDGID_array


def maxHistoryPDGID(id_array, mom_array, counts):
    if ak.backend(id_array) == "typetracer":
        ak.typetracer.length_zero_if_typetracer(id_array)
        ak.typetracer.length_zero_if_typetracer(mom_array)
        ak.typetracer.length_zero_if_typetracer(counts)
        return ak.Array(id_array.layout.to_typetracer(forget_length=True))
    return _maxHistoryPDGID(id_array, mom_array, counts)
