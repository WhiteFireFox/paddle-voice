import wave
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

path = 'wave/wav/'
dir1 = os.listdir(path)

# 打开WAV文档
for i in range(len(glob.glob(pathname='wave/wav/*.wav'))):
    f = wave.open(path + dir1[i])
    params = f.getparams()
    print(params)

    # params中前四位分别为：nchannels:声道数、sampwidth:量化位数、framerate:采样频率、nframes:采样点数
    framerate = params[2]
    nframes = params[3]

    # 读取波形数据
    str_data = f.readframes(nframes)
    f.close()

    # 将波形数据转换为数组
    WaveData = np.fromstring(str_data, dtype=np.short)
    WaveData.shape = -1, 2
    WaveData = WaveData.T
    time = np.arange(0, nframes) * (1.0 / framerate)

    # 绘制波形
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(time, WaveData[0])
    plt.show()

    # 保存数据集
    plt.axis('off')
    plt.savefig('wave/pic/'+str(i)+'.jpg')
    plt.clf()
