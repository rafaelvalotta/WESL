import numpy as np
from pathlib import Path

path = Path("/Users/brunoboer/Downloads/sgd_data/square_runs/rows_10_det/seed_1/")
for f in sorted(path.glob('det_*.npy')):
    data = np.load(f)
    print(f"{f.name:12} shape={data.shape}")
    
    if data.ndim == 1:  # sgd_aep, con, time
        print(f"  head (0:2):  {data[:2]}")
        print(f"  meio (~250): {data[len(data)//2-1:len(data)//2+1] if len(data)>1 else 'N/A'}")
        print(f"  tail (-2:):  {data[-2:]}")
        print(f"  min/max:     {data.min():.2f} / {data.max():.2f}")

    else:  # det_x, det_y (500,100)
        print(f"  head shape:   {data[:2].shape}")
        print(f"  x[0,0:5]:     {data[0,:5]}")
        print(f"  meio shape:   {data[len(data)//2-1:len(data)//2+1].shape}")
        print(f"  x[250,0:5]:   {data[len(data)//2,:5]}")
        print(f"  tail shape:   {data[-2:].shape}")
        print(f"  x[-1,0:5]:    {data[-1,:5]}")
        
        # 🆕 DETALHE FINAL: min/max da ÚLTIMA LINHA
        final_line = data[-1, :]
        print(f"  🆕 x[-1] min/max:  {final_line.min():6.2f} / {final_line.max():6.2f}m")
        print(f"  🆕 # x[-1]<0:       {np.sum(final_line < 0)} turbinas")
        print(f"  🆕 # x[-1]>3620:    {np.sum(final_line > 3620)} turbinas")
        
        # 🆕 Range GLOBAL vs FINAL
        print(f"  x global:     {data.min():4.0f} - {data.max():4.0f}m (500 iters)")
    
    print()
