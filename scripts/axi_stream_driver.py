from datetime import datetime
import numpy as np
from pynq import Overlay, Xlnk

class NeuralNetworkOverlay(Overlay):
    def __init__(
        self, bitfile_name, x_shape, y_shape,
        dtype=np.float32
    ):
        super().__init__(
            bitfile_name,
        )
        # DMA channels
        timea = datetime.now() # Timing inference

        self.sendchannel = self.hier_0.axi_dma_0.sendchannel
        self.recvchannel = self.hier_0.axi_dma_0.recvchannel

        timeb = datetime.now()
        print((timeb - timea).total_seconds())
        # CMA allocator (PYNQ 2.4)
        self.xlnk = Xlnk()
        self.input_buffer = self.xlnk.cma_array(shape=x_shape, dtype=dtype)
        self.output_buffer = self.xlnk.cma_array(shape=y_shape, dtype=dtype)
    def _print_dt(self, timea, timeb, N):
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 1e-6
        rate = N / dts
        print(f"Classified {N} samples in {dts} seconds ({rate} inferences / s)")
        return dts, rate
    def predict(self, X, debug=False, profile=False, encode=None, decode=None):
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)
        self.input_buffer[:] = X
        self.sendchannel.transfer(self.input_buffer)
        self.recvchannel.transfer(self.output_buffer)
        if debug:
            print("Transfer started")
        self.sendchannel.wait()
        self.recvchannel.wait()
        if decode is not None:
            return decode(self.output_buffer)
        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            return self.output_buffer, dts, rate
        return self.output_buffer
