# encoding = utf-8
import pandas as pd
from reader import *
import os
from concurrent.futures import ProcessPoolExecutor
import time


MARKER1 = [185, 115, 80, 65, 50, 30, 25, 15, 10]    # 通用marker表

def get_analysis(filename):
    print(f"{filename} start")
    gel = pre_cut(filename, True)
    lanes = gel_crop(gel)
    print(f"{filename} complete")


if __name__ == "__main__":
    files = os.listdir(os.getcwd())
    pics = []
    for f in files:
        if ".jpg" in f:
            pics.append(f)
    
    t1 = time.time()
    with ProcessPoolExecutor(10) as pool:
        for p in pics:
            pool.submit(get_analysis, p)
    print(time.time()-t1)
    # out = pd.read_excel("output.xlsx")
    # row = 0
    # for p in pic:
    #     try:
    #         gel = pre_cut(p, True)
    #         lanes = gel_crop(gel)
    #         mwf = normalize(lanes[7], MARKER1)
    #         for i in range(8, 14):
    #             peaks, areas = intensiy_integrate(lanes[i])
    #             out.iloc[row,3] = "{:.2f}".format(band_purity(areas,out["蛋白类型"][row]) * 100)
    #             row += 1
    #     except Exception as e:
    #         print(e)
    # out.to_excel("output.xlsx", index=False)