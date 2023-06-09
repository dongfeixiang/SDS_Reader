# encoding = utf-8
from reader import *


MARKER1 = [185, 115, 80, 65, 50, 30, 25, 15, 10]    # 通用marker表


if __name__ == "__main__":
    try:
        gel = pre_cut("3.jpg",True)
        lanes = gel_crop(gel)
        mwf = normalize(lanes[7], MARKER1)
        for i in range(8, 14):
            peaks, areas = intensiy_integrate(lanes[i])
            print(peaks, areas)
            print(band_purity(areas,1))
    except Exception as e:
        print(e)