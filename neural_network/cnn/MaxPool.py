# Do maxpool operation
# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: MaxPoo.py

import numpy as np

class MaxPool:
    def iter_region(self, img):
        h, w, _ = img.shape

        nh = h // 2
        nw = w // 2

        for i in range(nh):
            for j in range(nw):
                region = img[i * 2: i * 2 + 2, j * 2: j * 2 + 2]

                yield region, i, j

    def forward(self, input):
        h, w, num = input.shape

        self.input = input

        output = np.zeros((h // 2, w // 2, num))

        for region, i, j in self.iter_region(input):
            output[i, j] = np.amax(region, axis = (0, 1))

        return output

    def backprop(self, dL_dout):
        dL_dinput = np.zeros(self.input.shape)

        for region, i, j in self.iter_region(self.input):
            h, w, f = region.shape

            amax = np.amax(region, axis = (0, 1))

            for ii in range(h):
                for jj in range(w):
                    for k in range(f):
                        if (region[ii, jj, k] == amax[k]):
                            dL_dinput[i * 2 + ii, j * 2 + jj, k] = dL_dout[i, j, k]

        return dL_dinput
