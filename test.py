# encoding = utf-8
import pandas as pd
from reader import *


MARKER1 = [185, 115, 80, 65, 50, 30, 25, 15, 10]    # 通用marker表


if __name__ == "__main__":
    pic = ["1.jpg", "2.jpg"]
    out = pd.read_excel("output.xlsx")
    row = 0
    for p in pic:
        try:
            gel = pre_cut(p, True)
            lanes = gel_crop(gel)
            mwf = normalize(lanes[7], MARKER1)
            for i in range(8, 14):
                peaks, areas = intensiy_integrate(lanes[i])
                out.iloc[row,3] = "{:.2f}".format(band_purity(areas,out["蛋白类型"][row]) * 100)
                row += 1
        except Exception as e:
            print(e)
    out.to_excel("output.xlsx", index=False)